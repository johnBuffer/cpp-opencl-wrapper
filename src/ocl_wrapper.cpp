#include "ocl_wrapper.hpp"


const std::string& oclw::getErrorString(cl_int err_num)
{
	const int32_t error_code = -(err_num < -19 ? err_num + 10 : err_num);
	return oclw::cl_errors[error_code];
}


const std::string oclw::loadSourceFromFile(const std::string& filename)
{
	std::ifstream kernel_file(filename, std::ios::in);
	if (!kernel_file.is_open()) {
		throw Exception(-1, "Cannot open source file '" + filename + "'");
	}
	std::ostringstream oss;
	oss << kernel_file.rdbuf();

	return oss.str();
}


void oclw::checkError(cl_int err_num, const std::string& err_message)
{
	if (err_num != CL_SUCCESS) {
		throw oclw::Exception(err_num, err_message + " [" + getErrorString(err_num) + "]");
	}
}
