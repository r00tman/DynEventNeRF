#!/usr/bin/env python3
import dv_processing as dvp
import dvpstat_cpp

def getCameraNameByCameraId(infn, cameraIdx):
    # follow exactly what edi_cpp does
    rof = dvpstat_cpp.ReadOnlyFile(infn)
    idx = 0
    lastCameraName = None
    for stream in rof.getFileInfo().mStreams:
        cameraName = stream.getSource()
        if cameraName == lastCameraName:
            continue
        lastCameraName = cameraName
        print(f"name={cameraName} currentIdx={idx} requestedIdx={cameraIdx}")

        if idx == cameraIdx:
            print(f'found camera {cameraName} at idx={idx}')
            return cameraName

        idx += 1
    return None


def getStreamByCameraId(infn, cameraIdx):
    cameraName = getCameraNameByCameraId(infn, cameraIdx)

    mcr = dvp.io.MonoCameraRecording(infn, cameraName)
    # assert mcr.isEventStreamAvailable()
    return mcr
