// mv_aedat_player: aedat4 multi-view player
#include <dv-processing/core/frame.hpp>
#include <dv-processing/io/mono_camera_recording.hpp>

#include <CLI/CLI.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <csignal>
#include <iostream>
#include <string>

static std::atomic<bool> globalShutdown(false);

static void handleShutdown(int) {
	globalShutdown.store(true);
}

int main(int ac, char **av) {
	using namespace std::chrono_literals;

	std::string aedat4Path = "/home/work/research/rec/23-12-04/dvSaveExt-calib-2023_12_04_18_52_55.aedat4";

	// Use CLI11 library to handle argument parsing
	CLI::App app{"Command-line aedat4 preview player of recorded frames and events"};

	app.add_option("-i,--input", aedat4Path, "Path to an input aedat4 file to be played.")->default_str(aedat4Path);

/*     dv::io::MonoCameraRecording reader("/home/work/research/rec/23-12-04/dvSaveExt-calib-2023_12_04_18_52_55.aedat4"); */
	try {
		app.parse(ac, av);
	}
	catch (const CLI::ParseError &e) {
		return app.exit(e);
	}

	// Install signal handlers for a clean shutdown
	std::signal(SIGINT, handleShutdown);
	std::signal(SIGTERM, handleShutdown);

	std::vector<dv::io::MonoCameraRecording> readers;

	// Construct the reader
	dv::io::ReadOnlyFile file(aedat4Path);
	dv::io::FileInfo info = file.getFileInfo();
	std::string lastCameraName;
	for(const auto &stream: info.mStreams) {
		const std::string cameraName = stream.getSource().value();
		if (cameraName == lastCameraName) {
			continue;
		}
		lastCameraName = cameraName;
		std::cout << cameraName << std::endl;
		dv::io::MonoCameraRecording reader(aedat4Path, cameraName);

		// Placeholders for stream names
		std::string frameStream;
		std::string eventStream;

		// Find streams with compatible types from the list of all available streams
		for (const auto &name : reader.getStreamNames()) {
			if (reader.isStreamOfDataType<dv::Frame>(name) && frameStream.empty()) {
				frameStream = name;
			}
			else if (reader.isStreamOfDataType<dv::EventPacket>(name) && eventStream.empty()) {
				eventStream = name;
			}
		}

		// Named variables to hold the availability of streams
		const bool framesAvailable = !frameStream.empty();
		const bool eventsAvailable = !eventStream.empty();

		// Check whether at least one of the streams is available
		if (!framesAvailable && !eventsAvailable) {
			throw dv::exceptions::InvalidArgument<dv::cstring>(
				"Aedat4 player requires a file with at least event or frame stream available", aedat4Path);
		}

		// Create a display windows for both types
		if (framesAvailable) {
			auto winName = "AEDAT4 Player - Frames - "+cameraName;
			cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
			// cv::namedWindow("AEDAT4 Player - Frames", cv::WINDOW_GUI_NORMAL);
			auto idx = readers.size();
			cv::moveWindow(winName, idx%3*400, idx/3*320);
		}
		if (eventsAvailable) {
			// cv::namedWindow("AEDAT4 Player - Events - "+cameraName, cv::WINDOW_AUTOSIZE);
		}

		readers.emplace_back(reader);
	}




	// Buffer to store last frame
	std::vector<dv::Frame> lastFrames(readers.size());
	for (auto &lastFrame: lastFrames) {
		lastFrame.timestamp = -1;
	}
	int64_t lastTimestamp = -1;

	std::vector<std::tuple<int64_t, int> > newTimestamps(readers.size());
	std::vector<dv::Frame> newFrames(readers.size());

	// // Declare accumulator; Use pointer type in case event stream is unavailable
	// std::unique_ptr<dv::Accumulator> accumulator;

	bool isRunning = true;
	for (const auto &reader: readers) {
		isRunning = isRunning && reader.isRunning();
	}


	// Main reading loop
	while (!globalShutdown && isRunning) {
		isRunning = true;
		// for (const int i: {5, 4, 0, 2, 3, 1}) {
		for (int i = 0; i < readers.size(); ++i) {
			auto &reader = readers[i];
			auto &lastFrame = lastFrames[i];
			// Read frames if available
			if (auto frame = reader.getNextFrame(); frame.has_value()) {
				newFrames[i] = *frame;
				newTimestamps[i] = {frame->timestamp, i};
				// Buffer the first frame
				// if (lastFrame.timestamp > 0) {
					// Show the frame
					// cv::imshow("AEDAT4 Player - Frames - "+reader.getCameraName(), lastFrame.image);

					// Calculate sleep duration between frames, set to a minimum value of 1, a zero will
					// freeze the GUI
					// auto sleep = static_cast<int>((frame->timestamp - lastTimestamp) / 1000LL);
					// auto sleep = static_cast<int>((newFrames[i]->timestamp - lastTimestamp) / 1000LL);
					// std::cout << i << " " << sleep << " " << newFrames[i]->timestamp << " " << lastTimestamp << std::endl;
					// sleep = std::max(sleep, 1);

					// Sleep until next frame timestamp
					// cv::waitKey(sleep);
				// }

				// Move the next frame for rendering in next iteration
				// lastFrame = *frame;
				// lastTimestamp = lastFrame.timestamp;
			}

			// Read events if available
			// Read event in a loop, this is needed since events are stored in small batches of short period of time
			while (const auto events = reader.getNextEventBatch()) {
				// Read until we get in front of the latest frame; In case no frames are available, the loop will
				// read one batch per main loop iteration.
				if (events->getHighestTime() > lastFrame.timestamp) {
					break;
				}
			}

			isRunning = isRunning && reader.isRunning();
		}
		std::sort(newTimestamps.begin(), newTimestamps.end());
		for (auto [ts, i]: newTimestamps) {
			auto &reader = readers[i];
			auto &lastFrame = lastFrames[i];
			auto &newFrame = newFrames[i];

            cv::Mat cvt;
            cv::cvtColor(newFrame.image, cvt, cv::COLOR_BayerBG2BGR);

			/* cv::imshow("AEDAT4 Player - Frames - "+reader.getCameraName(), newFrame.image); */
			cv::imshow("AEDAT4 Player - Frames - "+reader.getCameraName(), cvt);
			if (lastTimestamp >= 0) {
				auto sleep = static_cast<int>((newFrame.timestamp - lastTimestamp) / 1000LL);
				std::cout << i << " " << sleep << " " << newFrame.timestamp << " " << lastTimestamp << std::endl;
				// if (sleep > 0) {
				// 	cv::waitKey(sleep);
				// }
				if (sleep < 1) {
					sleep = 1;
				}
				cv::waitKey(sleep);
			}
			lastTimestamp = newFrame.timestamp;
		}
		std::swap(lastFrames, newFrames);
	}

	return EXIT_SUCCESS;
}
