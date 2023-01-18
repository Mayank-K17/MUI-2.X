
#include <mui.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <mpi4py/mpi4py.h>
#include <string>

#include "config_name.h"

std::string get_mpi_version()
{
#ifdef MPI_VERSION_STR
  return MPI_VERSION_STR;
#else
  return "";
#endif
}

std::string get_compiler_version()
{
#ifdef COMPILER_VERSION_STR
  return COMPILER_VERSION_STR;
#else
  return "";
#endif
}

std::string get_compiler_config()
{
#ifdef COMPILER_CONFIG_STR
  return COMPILER_CONFIG_STR;
#else
  return "";
#endif
}

template <class Tconfig>
void declare_create_uniface(py::module &m)
{
  std::string name = "_create_uniface" + config_name<Tconfig>();
  m.def(name.c_str(), [](std::string domain, std::vector<std::string> interfaces, py::handle world)
        { return mui::create_uniface<Tconfig>(domain, interfaces, MPI_COMM_WORLD); });
}

template <typename Tconfig, typename T, template <typename, typename, typename> class Tsampler>
std::string sampler_name()
{
  if (std::is_same<Tsampler<Tconfig, T, T>, mui::sampler_exact<Tconfig, T, T>>::value)
    return "exact";
  if (std::is_same<Tsampler<Tconfig, T, T>, mui::sampler_gauss<Tconfig, T, T>>::value)
    return "gauss";
  if (std::is_same<Tsampler<Tconfig, T, T>, mui::sampler_moving_average<Tconfig, T, T>>::value)
    return "moving_average";
  if (std::is_same<Tsampler<Tconfig, T, T>, mui::sampler_nearest_neighbor<Tconfig, T, T>>::value)
    return "nearest_neighbor";
  if (std::is_same<Tsampler<Tconfig, T, T>, mui::sampler_pseudo_n2_linear<Tconfig, T, T>>::value)
    return "pseudo_n2_linear";
  throw std::runtime_error("Invalid sampler type");
}

template <typename Tchrono>
std::string chrono_sampler_name()
{
  return "chrono_exact";
}

template <typename Tconfig, typename T, template <typename, typename, typename> class Tsampler, template <typename> class Tchrono>
void declare_uniface_fetch(py::class_<mui::uniface<Tconfig>> &uniface)
{
  using Tclass = mui::uniface<Tconfig>;
  using Treal = typename Tconfig::REAL;
  using Ttime = typename Tconfig::time_type;

  std::string fetch_name = "fetch_" + type_name<T>() + "_" + sampler_name<Tconfig, T, Tsampler>() + "_";
  uniface.def(fetch_name.c_str(),
              (T(Tclass::*)(
                  const std::string &, const mui::point<Treal, Tconfig::D> &, const Ttime,
                  const Tsampler<Tconfig, T, T> &,
                  const Tchrono<Tconfig> &, bool)) &
                  Tclass::fetch,
              "");
}

template <typename Tconfig, typename T, template <typename, typename, typename> class Tsampler>
void declare_uniface_fetch_all_chrono(py::class_<mui::uniface<Tconfig>> &uniface)
{
  declare_uniface_fetch<Tconfig, T, Tsampler, mui::chrono_sampler_exact>(uniface);
  if constexpr (!std::is_same_v<T, std::string>)
  {
    // Disable these chrono samplers for std::string
    declare_uniface_fetch<Tconfig, T, Tsampler, mui::chrono_sampler_gauss>(uniface);
    declare_uniface_fetch<Tconfig, T, Tsampler, mui::chrono_sampler_mean>(uniface);
    declare_uniface_fetch<Tconfig, T, Tsampler, mui::chrono_sampler_sum>(uniface);
  }
}

template <typename Tconfig, typename T>
void declare_uniface_funcs(py::class_<mui::uniface<Tconfig>> &uniface)
{
  using Tclass = mui::uniface<Tconfig>;
  using Treal = typename Tconfig::REAL;

  std::string push_name = "push_" + type_name<T>();
  std::string push_many_name = "push_many_" + type_name<T>();
  std::string fetch_name = "fetch_" + type_name<T>();
  uniface.def(push_name.c_str(),
              (void(Tclass::*)(const std::string &, const mui::point<Treal, Tconfig::D> &,
                               const T &)) &
                  Tclass::push,
              "")
      .def(push_name.c_str(),
           (void(Tclass::*)(const std::string &, const T &)) & Tclass::push,
           "")
      .def(fetch_name.c_str(),
           (T(Tclass::*)(const std::string &)) & Tclass::fetch, "");

  // Do not use std::string in the following templates (C++17)
  if constexpr (!std::is_same_v<T, std::string>)
  {
    uniface.def(push_many_name.c_str(),
                (void(Tclass::*)(const std::string &attr, const py::array_t<Treal> &points,
                                 const py::array_t<T> &values)) &
                    Tclass::push_many,
                "");
    declare_uniface_fetch_all_chrono<Tconfig, T, mui::sampler_gauss>(uniface);
  }

  declare_uniface_fetch_all_chrono<Tconfig, T, mui::sampler_exact>(uniface);
}

template <typename Tconfig>
void declare_uniface_class(py::module &m)
{
  std::string name = "_Uniface" + config_name<Tconfig>();
  using Tclass = mui::uniface<Tconfig>;
  using Treal = typename Tconfig::REAL;
  using Ttime = typename Tconfig::time_type;
  py::class_<Tclass> uniface(m, name.c_str());

  uniface
      .def("commit", (int(Tclass::*)(Ttime)) & Tclass::commit, "")
      .def("forecast", (void(Tclass::*)(Ttime)) & Tclass::forecast, "")
      //.def("is_ready", &Tclass::is_ready, "")
      .def("barrier", (void(Tclass::*)(Ttime)) & Tclass::barrier, "")
      .def("barrier", (void(Tclass::*)(Ttime, Ttime)) & Tclass::barrier, "")
      .def("forget", (void(Tclass::*)(Ttime, bool)) & Tclass::forget, "")
      .def("forget", (void(Tclass::*)(Ttime, Ttime, bool)) & Tclass::forget, "")
      .def("set_memory", (void(Tclass::*)(Ttime)) & Tclass::set_memory, "")
      .def("announce_send_span",
           (void(Tclass::*)(Ttime, Ttime, mui::geometry::any_shape<Tconfig>,
                            bool synchronised)) &
               Tclass::announce_send_span,
           "")
      .def("announce_recv_span",
           (void(Tclass::*)(Ttime, Ttime, mui::geometry::any_shape<Tconfig>,
                            bool synchronised)) &
               Tclass::announce_recv_span,
           "")
      .def("announce_send_disable",
           (void(Tclass::*)()) & Tclass::announce_send_disable, "")
      .def("announce_recv_disable",
           (void(Tclass::*)()) & Tclass::announce_recv_disable, "")
      //    DEFINE_MUI_UNIFACE_FETCH_5ARGS() DEFINE_MUI_UNIFACE_FETCH_6ARGS()

      .def(py::init<const std::string &>());

  declare_uniface_funcs<Tconfig, double>(uniface);
  declare_uniface_funcs<Tconfig, float>(uniface);
  declare_uniface_funcs<Tconfig, std::int32_t>(uniface);
  declare_uniface_funcs<Tconfig, std::int64_t>(uniface);
  declare_uniface_funcs<Tconfig, std::string>(uniface);
}

// Declaration of other files
void chrono_sampler(py::module &m);
void sampler(py::module &m);
void geometry(py::module &m);

PYBIND11_MODULE(mui4py_mod, m)
{
  m.doc() = "MUI bindings for Python."; // optional module docstring

  // Expose numerical limits from C++
  m.attr("numeric_limits_real") = std::numeric_limits<double>::min();
  m.attr("numeric_limits_int") = std::numeric_limits<int>::min();

  geometry(m);
  sampler(m);
  chrono_sampler(m);

#ifdef PYTHON_INT_64
  declare_uniface_class<mui::mui_config_1dx>(m);
  declare_uniface_class<mui::mui_config_2dx>(m);
  declare_uniface_class<mui::mui_config_3dx>(m);
  declare_uniface_class<mui::mui_config_1fx>(m);
  declare_uniface_class<mui::mui_config_2fx>(m);
  declare_uniface_class<mui::mui_config_3fx>(m);

  declare_create_uniface<mui::mui_config_1dx>(m);
  declare_create_uniface<mui::mui_config_2dx>(m);
  declare_create_uniface<mui::mui_config_3dx>(m);
  declare_create_uniface<mui::mui_config_1fx>(m);
  declare_create_uniface<mui::mui_config_2fx>(m);
  declare_create_uniface<mui::mui_config_3fx>(m);
#elif defined PYTHON_INT_32
  declare_uniface_class<mui::mui_config_1d>(m);
  declare_uniface_class<mui::mui_config_2d>(m);
  declare_uniface_class<mui::mui_config_3d>(m);
  declare_uniface_class<mui::mui_config_1f>(m);
  declare_uniface_class<mui::mui_config_2f>(m);
  declare_uniface_class<mui::mui_config_3f>(m);

  declare_create_uniface<mui::mui_config_1d>(m);
  declare_create_uniface<mui::mui_config_2d>(m);
  declare_create_uniface<mui::mui_config_3d>(m);
  declare_create_uniface<mui::mui_config_1f>(m);
  declare_create_uniface<mui::mui_config_2f>(m);
  declare_create_uniface<mui::mui_config_3f>(m);
#else
#error PYTHON_INT_[32|64] not defined.
#endif

  m.def("set_quiet", &mui::set_quiet, "");
  m.def(
      "mpi_split_by_app", []() -> py::handle
      {
    if (import_mpi4py() < 0)
      Py_RETURN_NONE;
    return PyMPIComm_New(mui::mpi_split_by_app()); },
      "");
  m.def("get_mpi_version", &get_mpi_version, "");
  m.def("get_compiler_config", &get_compiler_config, "");
  m.def("get_compiler_version", &get_compiler_version, "");
}
