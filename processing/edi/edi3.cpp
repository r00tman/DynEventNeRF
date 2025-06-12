// edi3: deblurring/interpolation/debayering/tonemapping of aedat4 using fast-edi
#include "opencv2/core.hpp"
#include <opencv2/core/hal/interface.h>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <dv-processing/core/core.hpp>
#include <dv-processing/core/frame.hpp>
#include <dv-processing/io/mono_camera_recording.hpp>

#include <CLI/CLI.hpp>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <queue>

#include <av/StreamWriter.hpp>
#include <av/Rational.hpp>



struct event_t {
    float t;
    int16_t x, y;
    int8_t p;  // 1, -1, and 0 for special cases
};

float C_pos = 1, C_neg = -1;
float EPS = 0;

struct accum_t {
    float t;
    int n_pos, n_neg;

    float get_I() {
        return exp(C_pos*n_pos + C_neg*n_neg);
    }

    float diff_C_pos() {
        return n_pos * get_I();
    }

    float diff_C_neg() {
        return n_neg * get_I();
    }
};

class Accumulator {
public:
    std::vector<accum_t> accums;
    float full_value = 0;

    float C_pos_grad = 0;
    float C_neg_grad = 0;

	float I0 = 0;
	bool I0_computed = false;

	bool isExposing = false;

    void process_event(const event_t &e) {
        if (accums.empty()) {
            assert(e.p == 0);
            accums.emplace_back(accum_t{e.t, 0, 0});
			isExposing = true;
            return;
        }

        accum_t last_accum = accums[accums.size()-1];

        accum_t new_accum = last_accum;
        new_accum.t = e.t;
        if (e.p > 0) {
            new_accum.n_pos += 1;
        } else if (e.p < 0) {
            new_accum.n_neg += 1;
        }

		if (isExposing) {
			full_value += (e.t - last_accum.t) * last_accum.get_I();
			C_pos_grad += (e.t - last_accum.t) * last_accum.diff_C_pos();
			C_neg_grad += (e.t - last_accum.t) * last_accum.diff_C_neg();

		}

		// shutter event: no longer exposing since this moment
		// accums still created for the purpose of frame synthesis,
		// but full_value (accumulated blurred value) and its gradients are
		// no longer updated
		if (e.p == 0) {
			isExposing = false;
		}

        accums.emplace_back(new_accum);
    }

    void recompute() {
        full_value = 0;
        C_pos_grad = 0;
        C_neg_grad = 0;

        for (int i = 1; i < accums.size(); ++i) {
            accum_t last_accum = accums[i-1];
            accum_t new_accum = accums[i];


            full_value += (new_accum.t - last_accum.t) * last_accum.get_I();
            C_pos_grad += (new_accum.t - last_accum.t) * last_accum.diff_C_pos();
            C_neg_grad += (new_accum.t - last_accum.t) * last_accum.diff_C_neg();

			if (new_accum.n_neg == last_accum.n_neg && new_accum.n_pos == last_accum.n_pos) {
				break;  // new_accum caused by a shutter event, thus no longer exposing, thus stop
			}
        }

		I0_computed = false;
    }

    void compute_I0(float B, float T0, float T1) {
        // B is the actual blurry value
        // full_value is the simulated blurry value, non-normalized
        // T0, T1 are the start and end times of the window generating B

        assert(abs(T0 - accums[0].t)<1e-6);
        assert(abs(T1 - accums[accums.size()-1].t)<1e-6);
        float result = B*(T1-T0)/full_value;
		assert(accums.size()>2 || abs(result-B)<1e-6); // keeps the original value if no events (accums)
		I0 = result;
		I0_computed = true;
        // return result;
    }

    float get_It(float t) {
		assert(I0_computed);
        // float I0 = get_I0(B, T0, T1);

        float It_norm = 0;
		bool found = false;

        for (int i = 0; i < accums.size(); ++i) {
            if (i+1 >= accums.size() || accums[i+1].t > t) {
                assert(accums[i].t <= t);
                It_norm = accums[i].get_I();
				found = true;
                break;
            }
        }

        return It_norm * I0;

    }

};

namespace av
{
template<typename... Args>
void writeLog(LogLevel level, internal::SourceLocation&& loc, std::string_view fmt, Args&&... args) noexcept
{
    std::cerr << loc.toString() << ": " << av::internal::format(fmt, std::forward<Args>(args)...) << std::endl;
}
void writeLog(LogLevel level, internal::SourceLocation&& loc, std::string msg) noexcept
{
    std::cerr << loc.toString() << ": " << msg << std::endl;
}
}// namespace av

template<typename Return>
Return assertExpected(av::Expected<Return>&& expected) noexcept
{
    if (!expected)
    {
        std::cerr << " === Expected failure == \n"
                  << expected.errorString() << std::endl;
        exit(EXIT_FAILURE);
    }

    if constexpr (std::is_same_v<Return, void>)
        return;
    else
        return expected.value();
}

av::Ptr<av::Frame> newWriteableVideoFrame(int width, int height, AVPixelFormat pix_fmt) {
		av::Ptr<av::Frame> f = av::makePtr<av::Frame>();
		AVFrame* frame = f->native();

		frame->width  = width;
		frame->height = height;
		frame->format = pix_fmt;
		frame->pts    = 0;

		/* allocate the buffers for the frame data */
		auto ret = av_frame_get_buffer(frame, 0);
		assert (ret >= 0);
		// if (ret < 0)
		// 	RETURN_AV_ERROR("Could not allocate frame data: {}", avErrorStr(ret));

		ret = av_frame_make_writable(frame);
		assert (ret >= 0);
		// if (ret < 0)
		// 	RETURN_AV_ERROR("Could not make frame writable: {}", avErrorStr(ret));

		return f;
}


struct MyVideoWriter {
	std::shared_ptr<av::StreamWriter> writer;
	int videoStreamIdx = -1;
	int width = -1, height = -1;
	AVPixelFormat pixFmt;

	std::shared_ptr<av::Frame> bufferFrame;

	void open(const std::string &path, int width_, int height_, int crf, double fps) {
		writer = av::StreamWriter::create(path).value();

		width = width_;
		height = height_;
		pixFmt = AV_PIX_FMT_RGB24;

		av::Rational frameRate(1.0 / fps, 100000);

		av::OptValueMap codecOpts = {{"crf", crf}};
		// todo: figure out how to set output pixFmt to yuv444p
		videoStreamIdx = writer->addVideoStream(AV_CODEC_ID_H264, width, height, pixFmt, frameRate, width, height, AV_PIX_FMT_YUV444P, std::move(codecOpts)).value();

		assertExpected(writer->open());
		bufferFrame = newWriteableVideoFrame(width, height, pixFmt);
	}

	void write(const cv::Mat &mat) {
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				auto linesize = bufferFrame->native()->linesize[0];
				uint8_t *out = &bufferFrame->native()->data[0][i*linesize+j*3+0];
				// out[0] = i%256;
				// out[1] = j%256;
				// out[2] = 0;
				// because of the BGR convention in OpenCV
				out[0] = mat.at<uint8_t>(i, j*3+2);
				out[1] = mat.at<uint8_t>(i, j*3+1);
				out[2] = mat.at<uint8_t>(i, j*3+0);
			}
		}
		assertExpected(writer->write(*bufferFrame, videoStreamIdx));
	}
	void close() {
		writer->flushAllStreams();
	}
};

class ColorSpace {
public:
	virtual float to_linear(float inp) = 0;
	virtual float from_linear(float inp) = 0;
};

class SRGBSpace : public ColorSpace {
	float to_linear(float inp) override {
		// expects inp in [0,1] range, outputs in [0,1] range
		const float thr = 0.04045;
		if (inp > thr) {
			return std::pow((inp+0.055)/1.055, 2.4);
		} else {
			return inp/12.92;
		}
	}

	float from_linear(float inp) override {
		// expects inp in [0,1] range, outputs in [0,1] range
		const float thr = 0.0031308;

		inp = std::max(inp, 0.0f); // so that pow doesn't return nans
		if (inp > thr) {
			return 1.055*std::pow(inp, 1/2.4)-0.055;
		} else {
			return inp*12.92;
		}
	}
};

class Gamma22Space : public ColorSpace {
	float to_linear(float inp) override {
		inp = std::max(inp, 0.0f); // so that pow doesn't return nans
		return std::pow(inp, 2.2f);
	}

	float from_linear(float inp) override {
		inp = std::max(inp, 0.0f); // so that pow doesn't return nans
		return std::pow(inp, 1/2.2f);
	}
};

class LinearSpace : public ColorSpace {
	float to_linear(float inp) override {
		return inp;
	}

	float from_linear(float inp) override {
		return inp;
	}
};

class LinearShiftSpace : public ColorSpace {
	float to_linear(float inp) override {
		return inp+EPS;
	}

	float from_linear(float inp) override {
		return inp-EPS;
	}
};

uint8_t float_to_uint8(float inp) {
	// converts [0,1] float into [0,255] uint8 with saturation
	inp = std::max<float>(std::min<float>(inp*255.f, 255.f), 0.f);
	return static_cast<uint8_t>(inp);
}

float uint8_to_float(uint8_t inp) {
	// converts [0,255] uint8 into [0,1] float
	return static_cast<float>(inp)/255.f;
}

void debayerImage(const cv::Mat &image, cv::Mat &out, cv::ColorConversionCodes code) {
	if (code < 0) {
		image.copyTo(out);
		return;
	}
	cv::cvtColor(image, out, code);
	if (code == cv::COLOR_BayerBG2BGR_VNG) {
		// shift by 1
		cv::Mat out_o = out.clone();

		int shiftX = 0;
		int shiftY = 2;
		for (int y = 0; y < image.rows; ++y) {
			for (int x = 0; x < image.cols; ++x) {
				int oldX = std::max(0, std::min(image.cols-1, x - shiftX));
				int oldY = std::max(0, std::min(image.rows-1, y - shiftY));

				out.at<cv::Vec3b>(y, x) = out_o.at<cv::Vec3b>(oldY, oldX);
			}
		}
	}
}

static std::atomic<bool> globalShutdown(false);

static void handleShutdown(int) {
	globalShutdown.store(true);
}

int main(int argc, char **argv) {
	using namespace std::chrono_literals;

	std::string aedat4Path = "/home/work/research/rec/23-12-04/dvSaveExt-calib-2023_12_04_18_52_55.aedat4";

	std::string outFP = "out.mp4";

	float startT = 0;
	float endT = 500;

	int64_t t0 = -1;

	float synthFps = 100;

	size_t cameraIdx = 0;
	bool dry_run = false;
	bool detect_t0_and_quit = false;

	std::string colorSpaceName = "linear";

	// Use CLI11 library to handle argument parsing
	CLI::App app{"Command-line aedat4 preview player of recorded frames and events"};

	app.add_option("-i,--input", aedat4Path, "Path to an input aedat4 file to be played.")->capture_default_str();
	app.add_option("-c,--camera", cameraIdx, "Camera idx to use from the input file")->capture_default_str();

	app.add_option("--t0", t0, "Absolute timestamp of the start (-1 for auto-detect)")->capture_default_str();
	app.add_option("-a,--start", startT, "Time of start for synthesized frames")->capture_default_str();
	app.add_option("-b,--end", endT, "Time of end for synthesized frames")->capture_default_str();
	app.add_option("-r,--fps", synthFps, "FPS of synthesized frames")->capture_default_str();

	app.add_option("-p,--cpos", C_pos, "Event generation positive threshold")->capture_default_str();
	app.add_option("-n,--cneg", C_neg, "Event generation negative threshold")->capture_default_str();
	app.add_option("-e,--eps", EPS, "Dark regions eps in [0,1]")->capture_default_str();
	app.add_option("-s,--colorspace", colorSpaceName, "Color space name (srgb, gamma22, linear, linearshift)")->capture_default_str();

	std::string debayeringName = "bilinear";
	app.add_option("--debayering", debayeringName, "Debayering Method (bilinear, vng, edgeaware, none)")->capture_default_str();
	app.add_option("-o,--output", outFP, "Output video file path")->capture_default_str();
	app.add_flag("-d,--dry_run", dry_run, "Dry run (load data and quit)")->capture_default_str();
	app.add_flag("--detect_t0", detect_t0_and_quit, "Dry run, detect t0 and quit")->capture_default_str();

	try {
		app.parse(argc, argv);
	}
	catch (const CLI::ParseError &e) {
		return app.exit(e);
	}
	// C_neg = -C_pos;

	// Install signal handlers for a clean shutdown
	std::signal(SIGINT, handleShutdown);
	std::signal(SIGTERM, handleShutdown);

	std::unique_ptr<ColorSpace> colorSpace;
	if (colorSpaceName == "srgb") {
		colorSpace = std::make_unique<SRGBSpace>();
	} else if (colorSpaceName == "gamma22") {
		colorSpace = std::make_unique<Gamma22Space>();
	} else if (colorSpaceName == "linear") {
		colorSpace = std::make_unique<LinearSpace>();
	} else if (colorSpaceName == "linearshift") {
		colorSpace = std::make_unique<LinearShiftSpace>();
	} else {
		throw dv::exceptions::InvalidArgument<dv::cstring>(
			"Unknown color space", colorSpaceName);
	}

	auto debayeringMethod = cv::COLOR_BayerBG2BGR;
	if (debayeringName == "bilinear") {
		debayeringMethod = cv::COLOR_BayerBG2BGR;
	} else if (debayeringName == "vng") {
		debayeringMethod = cv::COLOR_BayerBG2BGR_VNG;
	} else if (debayeringName == "edgeaware") {
		debayeringMethod = cv::COLOR_BayerBG2BGR_EA;
	} else if (debayeringName == "none") {
		/* debayeringMethod = static_cast<cv::ColorConversionCodes>(-1); */
		debayeringMethod = cv::COLOR_GRAY2BGR;
	} else {
		throw dv::exceptions::InvalidArgument<dv::cstring>(
			"Unknown debayering method", debayeringName);
	}

    std::unique_ptr<dv::io::MonoCameraRecording> reader;

	// Construct the reader
	dv::io::ReadOnlyFile file(aedat4Path);
	dv::io::FileInfo info = file.getFileInfo();
	std::string lastCameraName;
	size_t currentIdx = 0;
	for(const auto &stream: info.mStreams) {
		const std::string cameraName = stream.getSource().value();
		if (cameraName == lastCameraName) {
			continue;
		}
		lastCameraName = cameraName;
		std::cout << "name=" << cameraName << " currentIdx=" << currentIdx << " requestedIdx=" << cameraIdx << std::endl;
		if (currentIdx == cameraIdx) {
			reader = std::make_unique<dv::io::MonoCameraRecording>(aedat4Path, cameraName);

			// Placeholders for stream names
			std::string frameStream;
			std::string eventStream;

			// Find streams with compatible types from the list of all available streams
			for (const auto &name : reader->getStreamNames()) {
				if (reader->isStreamOfDataType<dv::Frame>(name) && frameStream.empty()) {
					frameStream = name;
				}
				else if (reader->isStreamOfDataType<dv::EventPacket>(name) && eventStream.empty()) {
					eventStream = name;
				}
			}

			// Named variables to hold the availability of streams
			const bool framesAvailable = !frameStream.empty();
			const bool eventsAvailable = !eventStream.empty();

			// Check whether at least one of the streams is available
			if (!framesAvailable || !eventsAvailable) {
				throw dv::exceptions::InvalidArgument<dv::cstring>(
					"Aedat4 player requires a file with at least event and frame stream available", aedat4Path);
			}
			break;
		}
		currentIdx++;
	}


	dv::Frame newFrame;
	std::vector<event_t> allEvents;
	std::vector<dv::Frame> allFrames;

	bool isRunning = true;
    isRunning = isRunning && reader->isRunning();

	// int64_t t0 = -1;

	struct timeline_t {
		float time;
		enum {
			EVENT,
			EXPOSURE_START,
			EXPOSURE_END,
			FRAME_SYNTH
		} type;
		size_t frame_id;
		size_t event_id;
	};
	std::vector<timeline_t> timeline;

	for (int i = 0; i < (endT-startT)*synthFps; ++i) {
		float t = startT+i/synthFps;
		timeline.emplace_back(timeline_t{t, timeline_t::FRAME_SYNTH, 0, 0});
	}

	// Main reading loop
	while (!globalShutdown && isRunning) {
		isRunning = true;

        // Read frames if available
        if (auto frame = reader->getNextFrame(); frame.has_value()) {
            newFrame = *frame;
			allFrames.push_back(*frame);
			if (t0 < 0) {
				t0 = frame->timestamp;
				std::cout << "detected t0=" << t0 << " (first frame)" << std::endl;
				if (detect_t0_and_quit) {
					return 0;
				}
			}
			int64_t exposureBegin = frame->timestamp - t0;
			int64_t exposureEnd = exposureBegin + frame->exposure.count();
			// std::cout << 'f' << ' ' << exposureBegin << ' ' << exposureEnd << ' ' << frame->exposure << std::endl;
			if (exposureBegin > startT * 1e6f && exposureEnd < endT * 1e6f) {
				timeline.push_back({
					exposureBegin/1e6f,
					timeline_t::EXPOSURE_START,
					allFrames.size()-1,
					0
				});
				timeline.push_back({
					exposureEnd/1e6f,
					timeline_t::EXPOSURE_END,
					allFrames.size()-1,
					0
				});
			}
        }

        // Read events if available
        // Read event in a loop, this is needed since events are stored in small batches of short period of time
        while (const auto events = reader->getNextEventBatch()) {
            // Read until we get in front of the latest frame; In case no frames are available, the loop will
            // read one batch per main loop iteration.
			// allEvents.add(*events);
			for (auto ev: *events) {
				int64_t ts = ev.timestamp();
				if (t0 < 0) {
					t0 = ts;
					std::cout << "detected t0=" << t0 << " (first event)" << std::endl;
					if (detect_t0_and_quit) {
						return 0;
					}
				}
				float ts_float = (ts - t0)/1e6;
				int16_t x = ev.x();
				int16_t y = ev.y();
				int8_t p = ev.polarity()?1:-1;
				event_t evt{ts_float, x, y, p};
				allEvents.emplace_back(evt);
				if (ts_float > startT && ts_float < endT) {
					timeline.emplace_back(timeline_t{ts_float, timeline_t::EVENT, 0, allEvents.size()-1});
				}
			}
			// std::cout << 'e' << ' ' << events->getLowestTime() - t0 << ' ' << events->getHighestTime() - t0 << std::endl;
            if (events->getHighestTime() > newFrame.timestamp + newFrame.exposure.count()) {
                break;
            }

        }

        isRunning = isRunning && reader->isRunning();

	}
	if (!allFrames.empty()) {
		std::cout << "last frame:" << std::endl;
		std::cout << "exposure start at " << (allFrames.rbegin()->timestamp-t0)/1e6 << std::endl;
		std::cout << "exposure end at " << (allFrames.rbegin()->timestamp+newFrame.exposure.count()-t0)/1e6 << std::endl;
	}
	if (!allEvents.empty()) {
		std::cout << "last event: " << allEvents.rbegin()->t << std::endl;
	}
	if (dry_run || detect_t0_and_quit) {
		return 0;
	}
	std::sort(timeline.begin(), timeline.end(),
			  [&](const timeline_t &a, const timeline_t &b) { return a.time < b.time; });


	float accStart = -1;
	std::vector<std::vector<Accumulator> > acc;

	// C_pos = 1;
	// C_neg = -C_pos;

	cv::namedWindow("blurred", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("deblurred", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("synthres", cv::WINDOW_AUTOSIZE);

    const int W = 346;
    const int H = 260;
	cv::moveWindow("blurred", 0, 0);
	cv::moveWindow("deblurred", (int)(W*1.1), 0);
	cv::moveWindow("synthres", (int)(W*2.2), 0);
	bool isExposing = false;

	auto fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
	// auto fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
	// cv::VideoWriter vw_deblur("deblur.mp4", fourcc, 24, cv::Size(W, H));
	// cv::VideoWriter vw(outFP, fourcc, 24, cv::Size(W, H));
	MyVideoWriter vw;
	vw.open(outFP, W, H, 10, 24);

	MyVideoWriter vw_deblur;
	std::string deblurOutFP = outFP;
    size_t dotPosition = deblurOutFP.find_last_of('.');
    deblurOutFP.insert(dotPosition, "_deblur");
	vw_deblur.open(deblurOutFP, W, H, 10, 24);

	MyVideoWriter vw_blur;
	std::string blurOutFP = outFP;
    dotPosition = blurOutFP.find_last_of('.');
    blurOutFP.insert(dotPosition, "_blur");
	vw_blur.open(blurOutFP, W, H, 10, 24);

	// vw_deblur.set(cv::VIDEOWRITER_PROP_QUALITY, 90);
	// vw.set(cv::VIDEOWRITER_PROP_QUALITY, 90);

	std::queue<timeline_t> pending_imsynths;

	for (size_t i = 0; i < timeline.size(); ++i) {
		if (globalShutdown) {
			break;
		}
		const timeline_t &tl = timeline[i];
		if (tl.type == timeline_t::EXPOSURE_START) {
			while (!pending_imsynths.empty()) {
				const auto &imsynth = pending_imsynths.front();

				cv::Mat synthres(H, W, CV_8UC1);
				if (!acc.empty()) {
					for (size_t i = 0; i < H; ++i) {
						for (size_t j = 0; j < W; ++j) {
							const float It_linear = acc[i][j].get_It(imsynth.time);
							const float It = colorSpace->from_linear(It_linear);
							synthres.at<uint8_t>(i, j) = float_to_uint8(It);
							// synthres.at<uint8_t>(i, j) = (uint8_t)std::max<float>(0.f, std::min<float>(255.f, 255.f*It));
							// synthres.at<uint8_t>(i, j) = (uint8_t)std::max<float>(0.f, std::min<float>(255.f, 255.f*std::pow(It, 1/2.2f)));
							// synthres.at<uint8_t>(i, j) = (uint8_t)std::max(0.f, std::min(255.f, 255.f*It));
						}
					}
				}
				cv::Mat cvt;
				// cv::cvtColor(synthres, cvt, debayeringMethod);
				debayerImage(synthres, cvt, debayeringMethod);
				float lastAccstart = acc.empty()?-1:acc[0][0].accums[0].t;
				std::cout << "synth at t=" << imsynth.time << " last accstart " << lastAccstart << std::endl;
				vw.write(cvt);
				cv::imshow("synthres", cvt);

				pending_imsynths.pop();
			}

			isExposing = true;
			accStart = tl.time;
			acc.clear();

			for (int i = 0; i < H; ++i) {
				acc.push_back(std::vector<Accumulator>());
				for (int j = 0; j < W; ++j) {
					acc[i].push_back(Accumulator());
				}
			}
			for (int i = 0; i < H; ++i) {
				for (int j = 0; j < W; ++j) {
					acc[i][j].process_event(event_t{tl.time, 0, 0, 0});
				}
			}
		}
		// std::cout << tl.time << " " << tl.type << ' ' << isExposing << std::endl;
		if (tl.type == timeline_t::EVENT && !acc.empty()) {
			event_t ev = allEvents[tl.event_id];
			acc[ev.y][ev.x].process_event(ev);
		}

		if (tl.type == timeline_t::EXPOSURE_END) {
			assert(!acc.empty());
			isExposing = false;
			for (int i = 0; i < H; ++i) {
				for (int j = 0; j < W; ++j) {
					acc[i][j].process_event(event_t{tl.time, 0, 0, 0});
				}
			}
			const dv::Frame &frame = allFrames[tl.frame_id];
			for (size_t i = 0; i < H; ++i) {
				for (size_t j = 0; j < W; ++j) {
					/* const float B_nonlinear = std::max<float>(EPS, uint8_to_float(frame.image.at<uint8_t>(i, j))); */
					const float B_nonlinear = uint8_to_float(frame.image.at<uint8_t>(i, j));
					const float B = colorSpace->to_linear(B_nonlinear);
					// const float B = std::pow(std::max<float>(EPS, frame.image.at<uint8_t>(i, j)/255.f), 2.2f);
					// const float B = frame.image.at<uint8_t>(i, j)/255.f;
					const int64_t exposureBeginInt = frame.timestamp - t0;
					const int64_t exposureEndInt = exposureBeginInt + frame.exposure.count();
					const float expStart = exposureBeginInt/1e6f;
					const float expEnd = exposureEndInt/1e6f;
					acc[i][j].compute_I0(B, expStart, expEnd);
				}
			}
			cv::Mat deblurred(H, W, CV_8UC1);
			for (size_t i = 0; i < H; ++i) {
				for (size_t j = 0; j < W; ++j) {
					const float I0_linear = acc[i][j].I0;
					const float I0 = colorSpace->from_linear(I0_linear);
					deblurred.at<uint8_t>(i, j) = float_to_uint8(I0);
					// deblurred.at<uint8_t>(i, j) = (uint8_t)std::max<float>(0.f, std::min<float>(255.f, 255.f*I0));
					// deblurred.at<uint8_t>(i, j) = (uint8_t)std::max<float>(0.f, std::min<float>(255.f, 255.f*std::pow(I0, 1/2.2f)));
				}
			}
			cv::Mat cvt;
			// cv::cvtColor(frame.image, cvt, debayeringMethod);
			debayerImage(frame.image, cvt, debayeringMethod);
			cv::imshow("blurred", cvt);
			vw_blur.write(cvt);
			// cv::cvtColor(deblurred, cvt, debayeringMethod);
			debayerImage(deblurred, cvt, debayeringMethod);
			vw_deblur.write(cvt);
			cv::imshow("deblurred", cvt);

			if (cv::waitKey(1) == 'q') {
				globalShutdown = true;
			}
		}
		if (tl.type == timeline_t::FRAME_SYNTH) {
			pending_imsynths.push(tl);
		}
	}

	while (!pending_imsynths.empty()) {
		if (globalShutdown) {
			break;
		}
		const auto &imsynth = pending_imsynths.front();

		if (!acc.empty()) {
			cv::Mat synthres(H, W, CV_8UC1);
			for (size_t i = 0; i < H; ++i) {
				for (size_t j = 0; j < W; ++j) {
					const float It_linear = acc[i][j].get_It(imsynth.time);
					const float It = colorSpace->from_linear(It_linear);
					synthres.at<uint8_t>(i, j) = float_to_uint8(It);
					// synthres.at<uint8_t>(i, j) = (uint8_t)std::max<float>(0.f, std::min<float>(255.f, 255.f*It));
					// const float It = acc[i][j].get_It(imsynth.time);
					// synthres.at<uint8_t>(i, j) = (uint8_t)std::max<float>(0.f, std::min<float>(255.f, 255.f*std::pow(It, 1/2.2f)));
					// synthres.at<uint8_t>(i, j) = (uint8_t)std::max(0.f, std::min(255.f, 255.f*It));
				}
			}
			cv::Mat cvt;
			// cv::cvtColor(synthres, cvt, debayeringMethod);
			debayerImage(synthres, cvt, debayeringMethod);
			std::cout << "synth at t=" << imsynth.time << " last accstart" << acc[0][0].accums[0].t << std::endl;
			vw.write(cvt);
			cv::imshow("synthres", cvt);
			if (cv::waitKey(1) == 'q') {
				globalShutdown = true;
			}
		}

		pending_imsynths.pop();
	}

	cv::destroyWindow("blurred");
	// vw.release();
	// vw_deblur.release();

	return EXIT_SUCCESS;
}
