#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>
#include <dv-processing/core/core.hpp>
#include <dv-processing/core/frame.hpp>
#include <dv-processing/io/mono_camera_recording.hpp>

namespace py = pybind11;

class ReadOnlyFile: public dv::io::ReadOnlyFile {
public:
    ReadOnlyFile(const std::string &filePath):
        dv::io::ReadOnlyFile(filePath) {
    }
};

PYBIND11_MODULE(dvpstat_cpp, m) {
    py::class_<ReadOnlyFile>(m, "ReadOnlyFile")
        .def(py::init<const std::string &>())
        .def("getFileInfo", &ReadOnlyFile::getFileInfo, "Get File Info");

    py::class_<dv::io::FileInfo>(m, "FileInfo")
        .def(py::init<>())
        .def_readwrite("mFileSize", &dv::io::FileInfo::mFileSize)
        .def_readwrite("mCompression", &dv::io::FileInfo::mCompression)
        .def_readwrite("mDataTablePosition", &dv::io::FileInfo::mDataTablePosition)
        .def_readwrite("mDataTableSize", &dv::io::FileInfo::mDataTableSize)
        .def_readwrite("mDataTable", &dv::io::FileInfo::mDataTable)
        .def_readwrite("mTimeLowest", &dv::io::FileInfo::mTimeLowest)
        .def_readwrite("mTimeHighest", &dv::io::FileInfo::mTimeHighest)
        .def_readwrite("mTimeDifference", &dv::io::FileInfo::mTimeDifference)
        .def_readwrite("mTimeShift", &dv::io::FileInfo::mTimeShift)
        .def_readwrite("mStreams", &dv::io::FileInfo::mStreams)
        .def_readwrite("mPerStreamDataTables", &dv::io::FileInfo::mPerStreamDataTables);

    /* py::class_<dv::io::Stream>(m, "Stream") */
    /*     .def(py::init<>()) */
    /*     .def("getTypeDescription", &dv::io::Stream::getTypeDescription) */
    /*     .def("getModuleName", &dv::io::Stream::getModuleName) */
    /*     .def("getOutputName", &dv::io::Stream::getOutputName) */
    /*     .def("getSource", &dv::io::Stream::getSource) */
    /*     .def_readwrite("mId", &dv::io::Stream::mId) */
    /*     .def_readwrite("mName", &dv::io::Stream::mName) */
    /*     .def_readwrite("mTypeIdentifier", &dv::io::Stream::mTypeIdentifier); */
}
