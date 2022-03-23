#include <fstream>
#include "HLS/hls.h"

template<unsigned int DATAWIDTH, typename O_TYPE = ac_int<DATAWIDTH, false> >
void log_stream(std::string path, ihc::stream<ac_int<DATAWIDTH, false>> &input_stream, unsigned int reps)
{
	std::ofstream ofs(path);
	for(auto i=0; i<reps;i++)
	{
		ac_int<DATAWIDTH, false> val = input_stream.read();
        O_TYPE val_fp=*reinterpret_cast<O_TYPE*>(&val);
		ofs << val_fp << '\n';

	}
	ofs.close();
}

