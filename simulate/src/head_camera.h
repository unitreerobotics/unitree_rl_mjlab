// The R1 head camera in simulation: renders the head's views offscreen (EGL, its own thread),
// resamples them into the real camera's fisheye picture (OpenCV fisheye model, same size and
// intrinsics as on the robot) and publishes it as H.264 over RTSP through ffmpeg, where the
// robot's camera stream would be. With two eyes the pictures go one above the other, left eye on top
// (1280 wide as one eye: the Quest's WebRTC played 2560-wide side by side at about 1 fps).

#pragma once

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>

#include "depth_camera.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct HeadCameraConfig
{
    std::string model_path;          // opencv_fisheye JSON, as arena-r1-kit's r1/head_camera.json
    // Per eye, the MJCF cameras whose views together cover its fisheye: one eye, or two one above the other.
    std::vector<std::vector<std::string>> eyes;
    int view_size = 640;             // square render size per view, pixels
    int fps = 30;
    std::string url;                 // RTSP publish address
    std::string ffmpeg = "ffmpeg";
    DepthCameraConfig depth;         // the module's depth image, rendered in this thread's context
};

class HeadCamera
{
public:
    // lock_model returns the model and data to read while the lock is held.
    using Locker = std::function<std::unique_lock<std::recursive_mutex>(const mjModel*&, mjData*&)>;

    HeadCamera(HeadCameraConfig cfg, Locker locker) : cfg_(std::move(cfg)), locker_(std::move(locker)) {}

    void run(const std::atomic_int& exit)
    {
        std::signal(SIGPIPE, SIG_IGN);  // ffmpeg exiting must not end the simulator
        if (!load_fisheye()) return;
        // The context only once the model is loaded and the viewer's window stands: with the desktop
        // on the same GPU (PRIME "nvidia"), an EGL context made while GLFW sets up its own fails.
        for (;;) {
            const mjModel* m = nullptr;
            mjData* d = nullptr;
            if (exit.load()) return;
            if (locker_(m, d), m && d) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!make_gl()) return;

        const mjModel* built_for = nullptr;
        auto next = std::chrono::steady_clock::now();
        const auto period = std::chrono::microseconds(1000000 / cfg_.fps);
        // Resampling and publishing run on their own thread, one picture behind the rendering: the GPU
        // renders picture n+1 while the CPU resamples picture n, so the slower of the two sets the
        // frame rate instead of their sum. Each picture's own delay stays the same.
        std::thread worker([this] { work(); });

        while (!exit.load()) {
            std::this_thread::sleep_until(next);
            next += period;
            if (std::chrono::steady_clock::now() > next) next = std::chrono::steady_clock::now() + period;

            const auto t0 = std::chrono::steady_clock::now();
            bool depth_now = false;
            {
                const mjModel* m = nullptr;
                mjData* d = nullptr;
                auto lock = locker_(m, d);
                t_lock_ += std::chrono::steady_clock::now() - t0;
                if (!m || !d) continue;
                if (m != built_for) {
                    wait_idle();  // build() replaces the views the worker reads
                    if (!build(m)) { lock.unlock(); std::this_thread::sleep_for(std::chrono::seconds(1)); continue; }
                    built_for = m;
                    depth_.reset();
                    if (cfg_.depth.enabled) {
                        depth_ = std::make_unique<DepthCamera>(cfg_.depth);
                        if (!depth_->build(m)) depth_.reset();
                    }
                }
                for (size_t i = 0; i < views_.size(); ++i) {
                    mjv_updateScene(m, d, &opt_, nullptr, &views_[i].cam, mjCAT_ALL, &views_[i].scn);
                }
                depth_now = depth_ && depth_->due();
                if (depth_now) depth_->update_scene(m, d, &opt_);
            }
            const auto t1 = std::chrono::steady_clock::now();
            mjrRect vp = {0, 0, cfg_.view_size, cfg_.view_size};
            for (auto& v : views_) {
                mjr_render(vp, &v.scn, &con_);
                mjr_readPixels(v.rgb.data(), nullptr, vp, &con_);
            }
            if (depth_now) depth_->render(&con_);
            const auto t2 = std::chrono::steady_clock::now();
            {
                // Hand the views over: wait until the worker is done with the previous picture.
                std::unique_lock<std::mutex> lk(work_mtx_);
                work_cv_.wait(lk, [this] { return !job_; });
                for (auto& v : views_) v.rgb.swap(v.work);
                job_ = true;
            }
            work_cv_.notify_all();
            const auto t3 = std::chrono::steady_clock::now();
            t_scene_ += t1 - t0; t_render_ += t2 - t1; t_wait_ += t3 - t2;
            if (++frames_ % (5 * cfg_.fps) == 0) {
                const auto now = std::chrono::steady_clock::now();
                auto ms = [](std::chrono::steady_clock::duration x, long n) {
                    return std::chrono::duration<double, std::milli>(x).count() / std::max(n, 1L);
                };
                std::lock_guard<std::mutex> lk(work_mtx_);
                const long n = 5 * cfg_.fps;
                std::cout << "[head camera] " << n / std::chrono::duration<double>(now - stats_from_).count()
                          << " fps; per frame ms: lock " << ms(t_lock_, n) << " scene " << ms(t_scene_, n)
                          << " render " << ms(t_render_, n) << " wait " << ms(t_wait_, n) << " | worker: resample "
                          << ms(t_resample_, worked_) << " publish " << ms(t_publish_, worked_) << std::endl;
                t_lock_ = t_scene_ = t_render_ = t_wait_ = t_resample_ = t_publish_ = {};
                worked_ = 0;
                stats_from_ = now;
            }
        }
        {
            std::lock_guard<std::mutex> lk(work_mtx_);
            stop_ = true;
        }
        work_cv_.notify_all();
        worker.join();
        close_ffmpeg();
    }

private:
    struct View
    {
        int eye = 0;
        int camid = -1;
        mjvCamera cam;
        mjvScene scn;
        std::vector<unsigned char> rgb;   // rendered into
        std::vector<unsigned char> work;  // the previous picture, being resampled by the worker
    };

    // The worker thread: resamples and publishes each picture handed over by run().
    void work()
    {
        std::vector<unsigned char> out;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(work_mtx_);
                work_cv_.wait(lk, [this] { return job_ || stop_; });
                if (stop_) return;
            }
            out.resize(width_ * out_height() * 3);
            const auto t0 = std::chrono::steady_clock::now();
            resample(out);
            const auto t1 = std::chrono::steady_clock::now();
            publish(out);
            const auto t2 = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lk(work_mtx_);
                t_resample_ += t1 - t0; t_publish_ += t2 - t1;
                ++worked_;
                job_ = false;
            }
            work_cv_.notify_all();
        }
    }

    void wait_idle()
    {
        std::unique_lock<std::mutex> lk(work_mtx_);
        work_cv_.wait(lk, [this] { return !job_; });
    }
    // Per fisheye pixel: the view (-1: outside every view), the top-left source byte and bilinear weights (x256).
    struct Tap { int view; int offset; uint16_t ax, ay; };

    bool load_fisheye()
    {
        try {
            auto j = YAML::LoadFile(cfg_.model_path);
            if (j["model"].as<std::string>() != "opencv_fisheye") throw std::runtime_error("not opencv_fisheye");
            width_ = j["width"].as<int>();
            height_ = j["height"].as<int>();
            fx_ = j["fx"].as<double>(); fy_ = j["fy"].as<double>();
            cx_ = j["cx"].as<double>(); cy_ = j["cy"].as<double>();
            k_ = j["d"].as<std::vector<double>>();
            if (k_.size() != 4) throw std::runtime_error("d needs 4 coefficients");
        } catch (const std::exception& e) {
            std::cerr << "[head camera] " << cfg_.model_path << ": " << e.what() << "; camera off\n";
            return false;
        }
        return true;
    }

    // The NVIDIA GPU as an EGL device, if there is one: the default display is often the integrated GPU.
    EGLDisplay discrete_display()
    {
        auto query = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
        auto platform = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
        if (!query || !platform) return EGL_NO_DISPLAY;
        EGLDeviceEXT devices[8];
        EGLint n = 0;
        if (!query(8, devices, &n)) return EGL_NO_DISPLAY;
        for (EGLint i = 0; i < n; ++i) {
            EGLDisplay dpy = platform(EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
            if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, nullptr, nullptr)) continue;
            const char* vendor = eglQueryString(dpy, EGL_VENDOR);
            if (vendor && std::string(vendor).find("NVIDIA") != std::string::npos) return dpy;
            eglTerminate(dpy);
        }
        return EGL_NO_DISPLAY;
    }

    // An OpenGL context on dpy (initialised) with a 1x1 pbuffer, made current; false and a reason if not.
    bool make_context(EGLDisplay dpy, std::string& why)
    {
        const EGLint attr[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                               EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_NONE};
        EGLConfig conf;
        EGLint n = 0;
        if (!eglBindAPI(EGL_OPENGL_API) || !eglChooseConfig(dpy, attr, &conf, 1, &n) || n < 1) {
            why = "no OpenGL pbuffer config";
            return false;
        }
        auto error = [](const char* what) {
            char code[16];
            std::snprintf(code, sizeof code, "0x%04x", eglGetError());
            return std::string(what) + " (EGL error " + code + ")";
        };
        EGLContext ctx = eglCreateContext(dpy, conf, EGL_NO_CONTEXT, nullptr);
        if (ctx == EGL_NO_CONTEXT) {
            why = error("no context");
            return false;
        }
        // MuJoCo renders into its own framebuffer: no surface is needed where the driver allows it
        // (surfaceless, as MuJoCo's EGL backend); a 1x1 pbuffer elsewhere.
        EGLSurface surf = EGL_NO_SURFACE;
        if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
            const EGLint pb[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
            surf = eglCreatePbufferSurface(dpy, conf, pb);
            if (surf == EGL_NO_SURFACE || !eglMakeCurrent(dpy, surf, surf, ctx)) {
                why = error("neither surfaceless nor pbuffer");
                if (surf != EGL_NO_SURFACE) eglDestroySurface(dpy, surf);
                eglDestroyContext(dpy, ctx);
                return false;
            }
        }
        dpy_ = dpy; surf_ = surf; ctx_ = ctx;
        return true;
    }

    bool make_gl()
    {
        // The NVIDIA device first, then the default display (with the desktop itself on NVIDIA, as
        // in PRIME "nvidia" mode, only the latter may give a context).
        // If the default display is itself NVIDIA (desktop on NVIDIA), it goes first: on an RTX 4060
        // with driver 580, a context on the device display was made but mjr_makeContext then crashed
        // inside libnvidia-eglcore.
        std::string why;
        EGLDisplay def = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        bool def_nvidia = false;
        if (def != EGL_NO_DISPLAY && eglInitialize(def, nullptr, nullptr)) {
            const char* vendor = eglQueryString(def, EGL_VENDOR);
            def_nvidia = vendor && std::string(vendor).find("NVIDIA") != std::string::npos;
        }
        const auto order = def_nvidia ? std::vector<EGLDisplay>{def, discrete_display()}
                                      : std::vector<EGLDisplay>{discrete_display(), def};
        for (EGLDisplay dpy : order) {
            if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, nullptr, nullptr)) continue;
            std::string reason;
            if (make_context(dpy, reason)) break;
            why += std::string(why.empty() ? "" : "; ") + eglQueryString(dpy, EGL_VENDOR) + ": " + reason;
        }
        if (ctx_ == EGL_NO_CONTEXT) {
            std::cerr << "[head camera] no EGL context (" << (why.empty() ? "no display" : why) << "); camera off\n";
            return false;
        }
        mjv_defaultOption(&opt_);
        mjr_defaultContext(&con_);
        std::cout << "[head camera] EGL " << eglQueryString(dpy_, EGL_VENDOR) << std::endl;
        return true;
    }

    // Context, scenes and the fisheye lookup for a (new) model.
    bool build(const mjModel* m)
    {
        if (m->vis.global.offwidth < cfg_.view_size || m->vis.global.offheight < cfg_.view_size) {
            std::cerr << "[head camera] the model's offscreen buffer is smaller than view_size " << cfg_.view_size
                      << "; set <visual><global offwidth offheight> in the MJCF\n";
            return false;
        }
        for (auto& v : views_) mjv_freeScene(&v.scn);
        views_.clear();
        mjr_freeContext(&con_);
        mjr_makeContext(m, &con_, mjFONTSCALE_100);
        mjr_setBuffer(mjFB_OFFSCREEN, &con_);

        for (size_t eye = 0; eye < cfg_.eyes.size(); ++eye) for (const auto& name : cfg_.eyes[eye]) {
            int id = mj_name2id(m, mjOBJ_CAMERA, name.c_str());
            if (id < 0) {
                std::cerr << "[head camera] camera " << name << " not in the model\n";
                return false;
            }
            View v;
            v.eye = int(eye);
            v.camid = id;
            mjv_defaultCamera(&v.cam);
            v.cam.type = mjCAMERA_FIXED;
            v.cam.fixedcamid = id;
            mjv_defaultScene(&v.scn);
            mjv_makeScene(m, &v.scn, 10000);
            // Shadows and the floor's reflection cost most of the render time and show the operator nothing.
            v.scn.flags[mjRND_SHADOW] = 0;
            v.scn.flags[mjRND_REFLECTION] = 0;
            v.rgb.resize(cfg_.view_size * cfg_.view_size * 3);
            v.work.resize(v.rgb.size());
            views_.push_back(std::move(v));
        }
        // All views share one body: the fisheye's frame is that body's, x forward, y left, z up.
        build_lookup(m);
        std::cout << "[head camera] " << views_.size() << " views of " << cfg_.view_size << " px -> "
                  << cfg_.eyes.size() << " x fisheye " << width_ << "x" << height_ << " (" << width_ << "x"
                  << out_height() << ") at " << cfg_.fps << " fps, " << cfg_.url << std::endl;
        return true;
    }

    void build_lookup(const mjModel* m)
    {
        const int H = out_height();
        taps_.assign(width_ * H, Tap{-1, 0, 0, 0});
        const double half = cfg_.view_size / 2.0;
        std::vector<std::array<double, 9>> axes;
        std::vector<double> focal;
        for (auto& v : views_) {
            std::array<double, 9> R;
            mju_quat2Mat(R.data(), m->cam_quat + 4 * v.camid);  // columns: camera x, y, z in the body
            axes.push_back(R);
            focal.push_back(half / std::tan(m->cam_fovy[v.camid] * M_PI / 360.0));
        }
        int covered = 0;
        for (int eye = 0; eye < int(cfg_.eyes.size()); ++eye) for (int py = 0; py < height_; ++py) {
            for (int px = 0; px < width_; ++px) {
                double ray[3];
                fisheye_ray(px + 0.5, py + 0.5, ray);
                // OpenCV camera (x right, y down, z forward) to the body (x forward, y left, z up).
                const double b[3] = {ray[2], -ray[0], -ray[1]};
                int best = -1;
                double best_fwd = 0, bu = 0, bv = 0;
                for (size_t i = 0; i < views_.size(); ++i) {
                    if (views_[i].eye != eye) continue;
                    const auto& R = axes[i];
                    double cxv = b[0] * R[0] + b[1] * R[3] + b[2] * R[6];
                    double cyv = b[0] * R[1] + b[1] * R[4] + b[2] * R[7];
                    double fwd = -(b[0] * R[2] + b[1] * R[5] + b[2] * R[8]);  // cameras look along -z
                    if (fwd <= 1e-6) continue;
                    double u = half + focal[i] * cxv / fwd;
                    double v = half - focal[i] * cyv / fwd;  // from the top
                    if (u < 0 || v < 0 || u > cfg_.view_size - 1 || v > cfg_.view_size - 1) continue;
                    if (fwd > best_fwd) { best = int(i); best_fwd = fwd; bu = u; bv = v; }
                }
                if (best >= 0) {
                    // readPixels rows run bottom to top.
                    const double fu = std::min(bu, cfg_.view_size - 1.001);
                    const double fv = std::min(cfg_.view_size - 1 - bv, cfg_.view_size - 1.001);
                    const int x0 = int(fu), y0 = int(fv);
                    taps_[(eye * height_ + py) * width_ + px] = Tap{best, 3 * (y0 * cfg_.view_size + x0),
                                                  uint16_t((fu - x0) * 256), uint16_t((fv - y0) * 256)};
                    ++covered;
                }
            }
        }
        if (covered < width_ * H) {
            std::cout << "[head camera] " << (width_ * H - covered) << " fisheye pixels outside every view (black)"
                      << std::endl;
        }
    }

    // OpenCV fisheye: pixel to unit ray (x right, y down, z forward); Newton on theta_d = theta(1 + k1 t^2 + ...).
    void fisheye_ray(double u, double v, double* r) const
    {
        const double x = (u - cx_) / fx_, y = (v - cy_) / fy_;
        const double td = std::hypot(x, y);
        double t = td;
        for (int i = 0; i < 20; ++i) {
            const double t2 = t * t;
            const double f = t * (1 + t2 * (k_[0] + t2 * (k_[1] + t2 * (k_[2] + t2 * k_[3])))) - td;
            const double df = 1 + t2 * (3 * k_[0] + t2 * (5 * k_[1] + t2 * (7 * k_[2] + t2 * 9 * k_[3])));
            t -= f / df;
        }
        const double s = td > 1e-12 ? std::sin(t) / td : 1.0;
        r[0] = x * s; r[1] = y * s; r[2] = std::cos(t);
    }

    void resample(std::vector<unsigned char>& out) const
    {
        const int threads = std::max(2u, std::thread::hardware_concurrency() / 2), n = width_ * out_height();
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([this, &out, t, n, threads] { resample_rows(out, n * t / threads, n * (t + 1) / threads); });
        }
        for (auto& th : pool) th.join();
    }

    void resample_rows(std::vector<unsigned char>& out, int begin, int end) const
    {
        const int row = 3 * cfg_.view_size;
        for (int i = begin; i < end; ++i) {
            const Tap& t = taps_[i];
            unsigned char* o = &out[3 * i];
            if (t.view < 0) { o[0] = o[1] = o[2] = 0; continue; }
            const unsigned char* p = views_[t.view].work.data() + t.offset;
            const int ax = t.ax, ay = t.ay;
            for (int c = 0; c < 3; ++c) {
                const int top = p[c] * (256 - ax) + p[c + 3] * ax;
                const int bot = p[c + row] * (256 - ax) + p[c + row + 3] * ax;
                o[c] = static_cast<unsigned char>((top * (256 - ay) + bot * ay + 32768) >> 16);
            }
        }
    }

    void publish(const std::vector<unsigned char>& frame)
    {
        auto now = std::chrono::steady_clock::now();
        if (!pipe_) {
            if (now < retry_at_) return;
            retry_at_ = now + std::chrono::seconds(2);
            // Frames are stamped when they arrive, not at a nominal rate: when rendering falls behind
            // the nominal fps, nominal stamps run ahead of real time and the player's buffer (the
            // Quest's jitter buffer) grows into seconds of delay.
            const std::string cmd = cfg_.ffmpeg + " -hide_banner -loglevel error -use_wallclock_as_timestamps 1"
                " -f rawvideo -pix_fmt rgb24 -s " + std::to_string(width_) + "x" + std::to_string(out_height()) +
                " -i - -fps_mode passthrough -c:v libx264 -preset ultrafast -tune zerolatency -g " +
                std::to_string(cfg_.fps) + " -pix_fmt yuv420p -f rtsp -rtsp_transport tcp " + cfg_.url;
            pipe_ = popen(cmd.c_str(), "w");
            if (!pipe_) return;
            std::cout << "[head camera] publishing to " << cfg_.url << std::endl;
        }
        if (fwrite(frame.data(), 1, frame.size(), pipe_) != frame.size() || fflush(pipe_) != 0) {
            std::cerr << "[head camera] ffmpeg stopped (is mediamtx running?); retrying in 2 s\n";
            close_ffmpeg();
        }
    }

    int out_height() const { return height_ * int(cfg_.eyes.size()); }

    void close_ffmpeg()
    {
        if (pipe_) { pclose(pipe_); pipe_ = nullptr; }
    }

    HeadCameraConfig cfg_;
    Locker locker_;
    int width_ = 0, height_ = 0;
    double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
    std::vector<double> k_;
    EGLDisplay dpy_ = EGL_NO_DISPLAY;
    EGLSurface surf_ = EGL_NO_SURFACE;
    EGLContext ctx_ = EGL_NO_CONTEXT;
    mjvOption opt_;
    mjrContext con_;
    std::vector<View> views_;
    std::unique_ptr<DepthCamera> depth_;
    std::vector<Tap> taps_;
    FILE* pipe_ = nullptr;
    long frames_ = 0;
    std::chrono::steady_clock::duration t_lock_{}, t_scene_{}, t_render_{}, t_wait_{};
    std::chrono::steady_clock::time_point stats_from_ = std::chrono::steady_clock::now();
    // Shared with the worker, under work_mtx_.
    std::mutex work_mtx_;
    std::condition_variable work_cv_;
    bool job_ = false, stop_ = false;
    long worked_ = 0;
    std::chrono::steady_clock::duration t_resample_{}, t_publish_{};
    std::chrono::steady_clock::time_point retry_at_{};
};
