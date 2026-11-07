// Exact selected-column decode; one output byte, no N x L int64 temporaries.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <stdexcept>
namespace py=pybind11;
py::array_t<uint8_t> extract(py::array_t<uint8_t,py::array::c_style> packed,
                            py::array_t<int64_t,py::array::c_style> indices,
                            int threads) {
    if(packed.ndim()!=2||indices.ndim()!=1||threads<1||threads>64)
        throw std::invalid_argument("invalid packed extraction arguments");
    const int64_t n=packed.shape(0),stride=packed.shape(1),k=indices.shape(0);
    if(n<1||k<1)throw std::invalid_argument("empty packed extraction");
    const auto* ix=indices.data();
    for(int64_t j=0;j<k;++j)
        if(ix[j]<0||ix[j]/2>=stride)
            throw std::invalid_argument("selected column outside packed input");
    py::array_t<uint8_t> result({n,k});
    const auto* input=packed.data();auto* output=result.mutable_data();
    {
        py::gil_scoped_release release;
        #pragma omp parallel for num_threads(threads) schedule(static)
        for(int64_t row=0;row<n;++row)
            for(int64_t j=0;j<k;++j)
                output[row*k+j]=(input[row*stride+ix[j]/2]>>((ix[j]%2)*4))&15;
    }
    return result;
}
PYBIND11_MODULE(_packed_extract,m){m.def("extract",&extract);}
