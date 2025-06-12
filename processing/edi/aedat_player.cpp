// aedat_player: aedat4 single-view player
#include <dv-processing/core/frame.hpp>
#include <dv-processing/io/mono_camera_recording.hpp>

#include <CLI/CLI.hpp>
#include <opencv2/highgui.hpp>

#include <csignal>

static std::atomic<bool> globalShutdown(false);

static void handleShutdown(int) {
	globalShutdown.store(true);
}

int main(int ac, char **av) {
	using namespace std::chrono_literals;

	std::string aedat4Path;

	// Use CLI11 library to handle argument parsing
	CLI::App app{"Command-line aedat4 preview player of recorded frames and events"};

	app.add_option("-i,--input", aedat4Path, "Path to an input aedat4 file to be played.");
	try {
		app.parse(ac, av);
	}
	catch (const CLI::ParseError &e) {
		return app.exit(e);
	}

	// Install signal handlers for a clean shutdown
	std::signal(SIGINT, handleShutdown);
	std::signal(SIGTERM, handleShutdown);

	// Construct the reader
	dv::io::MonoCameraRecording reader(aedat4Path);

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
		cv::namedWindow("AEDAT4 Player - Frames", cv::WINDOW_AUTOSIZE);
	}
	if (eventsAvailable) {
		cv::namedWindow("AEDAT4 Player - Events", cv::WINDOW_AUTOSIZE);
	}

	// Buffer to store last frame
	dv::Frame lastFrame;
	lastFrame.timestamp = -1;

	// Declare accumulator; Use pointer type in case event stream is unavailable
	std::unique_ptr<dv::Accumulator> accumulator;

	// Initialize event stream slicer
	dv::EventStreamSlicer slicer;
	if (eventsAvailable) {
		// Create an accumulator for the reader
		accumulator = std::make_unique<dv::Accumulator>(*reader.getEventResolution());

		// Use a very small delay when running with frames and a longer delay when running without
		const int delay = framesAvailable ? 2 : 33;

		// Register a callback for the slicer which accumulates event and shows a preview window
		slicer.doEveryTimeInterval(40ms, [&accumulator, delay](const auto events) {
			// Pass event and generate a frame
			accumulator->accept(events);
			const auto frame = accumulator->generateFrame();

			// Show a preview of the accumulated frame and sleep for a short period
			cv::imshow("AEDAT4 Player - Events", frame.image);
			cv::waitKey(delay);
		});
	}

	// Main reading loop
	while (!globalShutdown && reader.isRunning()) {
		// Read frames if available
		if (framesAvailable) {
			if (const auto frame = reader.getNextFrame(frameStream); frame.has_value()) {
				// Buffer the first frame
				if (lastFrame.timestamp > 0) {
					// Show the frame
					cv::imshow("AEDAT4 Player - Frames", lastFrame.image);

					// Calculate sleep duration between frames, set to a minimum value of 1, a zero will
					// freeze the GUI
					auto sleep = std::max(static_cast<int>((frame->timestamp - lastFrame.timestamp) / 1000LL), 1);

					// Sleep until next frame timestamp
					cv::waitKey(sleep);
				}

				// Move the next frame for rendering in next iteration
				lastFrame = *frame;
			}
		}

		// Read events if available
		if (eventsAvailable) {
			// Read event in a loop, this is needed since events are stored in small batches of short period of time
			while (const auto events = reader.getNextEventBatch(eventStream)) {
				// Pass events to the slicer
				slicer.accept(*events);

				// Read until we get in front of the latest frame; In case no frames are available, the loop will
				// read one batch per main loop iteration.
				if (events->getHighestTime() > lastFrame.timestamp) {
					break;
				}
			}
		}
	}

	return EXIT_SUCCESS;
}
/* #include <dv-processing/io/mono_camera_recording.hpp> */
/* #include <opencv2/highgui.hpp> */

/* int main() { */
/*     // Open a file */
/*     dv::io::MonoCameraRecording reader("/home/work/research/rec/23-12-04/dvSaveExt-calib-2023_12_04_18_52_55.aedat4"); */

/*     // Initiate a preview window */
/*     cv::namedWindow("Preview", cv::WINDOW_NORMAL); */

/*     // Variable to store the previous frame timestamp for correct playback */
/*     std::optional<int64_t> lastTimestamp = std::nullopt; */

/*     // Run the loop while data is available */
/*     while (reader.isRunning()) { */
/*         // Read a frame, check whether it is correct. */
/*         // The method does not wait for frame arrive, it returns immediately with */
/*         // latest available frame or if no data is available, returns a `std::nullopt`. */
/*         if (const auto frame = reader.getNextFrame(); frame.has_value()) { */
/*             // Print information about received frame */
/*             std::cout << *frame << std::endl; */

/*             // Show a preview of the image */
/*             cv::imshow("Preview", frame->image); */

/*             // Calculate the delay between last and current frame, divide by 1000 to convert microseconds */
/*             // to milliseconds */
/*             const int delay = lastTimestamp.has_value() ? (frame->timestamp - *lastTimestamp) / 1000 : 2; */

/*             // Perform the sleep */
/*             cv::waitKey(delay); */

/*             // Store timestamp for the next frame */
/*             lastTimestamp = frame->timestamp; */
/*         } */
/*     } */

/*     return 0; */
/* } */
