// Exact finite-row max(abs(score)) without a three-column temporary.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <omp.h>
namespace py=pybind11;
using AD=py::array_t<double,py::array::c_style|py::array::forcecast>;
py::array_t<double> magnitudes(AD weights){
 auto w=weights.unchecked<2>();
 if(w.shape(1)!=3)throw std::runtime_error("three finite score columns required");
 py::array_t<double> values(w.shape(0));auto out=values.mutable_unchecked<1>();
 {
  py::gil_scoped_release release;
  #pragma omp parallel for schedule(static) num_threads(8)
  for(int64_t i=0;i<w.shape(0);i++)out(i)=std::max(std::abs(w(i,0)),std::max(std::abs(w(i,1)),std::abs(w(i,2))));
 }
 return values;
}
PYBIND11_MODULE(_msc_active_norm,m){m.def("magnitudes",&magnitudes);}
