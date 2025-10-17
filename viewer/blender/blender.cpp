// ======================================================================== //
// Copyright 2022++ Ingo Wald                                               //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

// (c) Milan Jaros, IT4Innovations

#include "hayStack/HayMaker.h"
#include "viewer/DataLoader.h"
#if HS_HAVE_CUDA
# include <cuda_runtime.h>
#endif

#include "renderengine_data.h"
#include "renderengine_tcp.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#define STB_IMAGE_IMPLEMENTATION 1
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include <iostream>
#include <fstream>

#if HS_MPI
#include <unistd.h>
#endif

namespace hs {

	double t_last_render;

	typedef enum {
		DPMODE_NOT_SPECIFIED,
		DPMODE_DATA_PARALLEL,
		DPMODE_DATA_REPLICATED
	} DPMode;

	struct FromCL {
		/*! data groups per rank. '0' means 'auto - use few as we can, as
			many as we have to fit for given number of ranks */
		int dpr = 0;
		/*! num data groups. '0' will become '1' by default, but allows us
			for specifying 'hasn't been specified', which in turn allows
			the mpi mode to set it to either '1' or '1 per rank' depending
			on choesn dpmode */
		int ndg = 0;

		bool forceSingleGPU = false;

		DPMode dpMode = DPMODE_NOT_SPECIFIED;

		/*! which color map to use for color mapping (if applicable) */
		int cmID = 0;

		bool mergeUnstructuredMeshes = false;
		bool useBackground = true;
		float ambientRadiance = .6f;
		std::string xfFileName = "";
		std::string outFileName = "hayStack.png";
		vec2i fbSize = { 800,600 };
		bool createHeadNode = false;
		int  numExtraDisplayRanks = 0;
		int  numFramesAccum = 1;
		int  spp = 1;
		bool verbose = true;
		struct {
			vec3f vp = vec3f(0.f);
			vec3f vi = vec3f(0.f);
			vec3f vu = vec3f(0.f, 1.f, 0.f);
			float fovy = 60.f;
		} camera;
		bool measure = 0;
		std::string envMapFileName;

		int port = 7000;
	};
	FromCL fromCL;

	// std::string FromCL::outFileName = "hayStack.png";
	// bool FromCL::measure = 0;
	// bool FromCL::verbose = true;

	inline bool verbose() { return fromCL.verbose; }

	void usage(const std::string& error = "")
	{
		std::cout << "./hs{Offline,Viewer,ViewerQT} ... <args>" << std::endl;
		std::cout << "w/ args:" << std::endl;
		std::cout << "-xf file.xf   ; specify transfer function" << std::endl;
		if (!error.empty())
			throw std::runtime_error("fatal error: " + error);
		exit(0);
	}

	inline mini::common::vec3f get3f(char** av, int& i)
	{
		float x = std::stof(av[++i]);
		float y = std::stof(av[++i]);
		float z = std::stof(av[++i]);
		return mini::common::vec3f(x, y, z);
	}

	size_t computeHashFromString(const char* s)
	{
		size_t hash = 0;
		size_t FNV_PRIME = 0x00000100000001b3ull;
		for (int i = 0; s[i] != 0; i++)
			hash = hash * FNV_PRIME ^ s[i];
		return hash;
	}

	std::vector<int> parseCommaSeparatedListOfInts(std::string s)
	{
		std::vector<std::string> tokens;
		int last = 0;
		while (true) {
			int pos = s.find(",");
			if (pos == s.npos) {
				tokens.push_back(s);
				break;
			}
			else {
				tokens.push_back(s.substr(0, pos));
				s = s.substr(pos + 1);
			}
		}

		std::vector<int> result;
		for (auto s : tokens)
			if (s != "")
				result.push_back(std::stoi(s));
		return result;
	}

	void initAllGPUs()
	{
#if __APPLE__
		// no cuda on mac
#else
		int numGPUs = 0;
		cudaGetDeviceCount(&numGPUs);
		std::cout << "#hs: found " << numGPUs
			<< " CUDA devices... initializing each one of them"
			<< "\n(we may use only some of them, but still...) " << std::endl;
		for (int i = 0; i < numGPUs; i++) {
			cudaSetDevice(i);
			cudaFree(0);
		}
		cudaSetDevice(0);
#endif
	}

#if HS_MPI
	void determineLocalProcessID(mpi::Comm& world, int& localRank, int& localSize)
	{
# if HS_FAKE_MPI
		localRank = 0;
		localSize = 1;
# else
		world.barrier();
		std::vector<char> hostName(10000);
		gethostname(hostName.data(), hostName.size());
		size_t hash = computeHashFromString(hostName.data());
		std::vector<size_t> allHostNames(world.size);
		MPI_Allgather(&hash, sizeof(hash), MPI_BYTE,
			allHostNames.data(), sizeof(hash), MPI_BYTE,
			world.comm);
		localRank = 0;
		localSize = 0;
		for (int i = 0; i < world.size; i++) {
			if (allHostNames[i] != hash) continue;
			localSize++;
			if (i < world.rank) localRank++;
		}

		for (int i = 0; i < world.size; i++) {
			world.barrier();
			if (i != world.rank) continue;
			std::cout << "#hs(" << world.rank << "): determined local rank/size as "
				<< localRank << "/" << localSize << std::endl;
		}
#endif
	}
#endif

	int getIntFromEnv(const char* varName, int fallback)
	{
		const char* var = getenv(varName);
		if (!var) return fallback;
		return std::stoi(var);
	}

	std::string getPhysicalString(int gpuID)
	{
#ifdef __APPLE__
		return "<cpu>";
#else
		cudaDeviceProp props;
		cudaError_t rc = cudaGetDeviceProperties(&props, gpuID);
		if (rc != cudaSuccess)
			throw std::runtime_error("could not query cuda Device properties");
		return "PCI:"
			+ std::to_string(props.pciDomainID) + "."
			+ std::to_string(props.pciBusID) + "."
			+ std::to_string(props.pciDeviceID);
#endif
	}

	std::vector<int> selectGPUs(mpi::Comm& world, int localRank, int localSize)
	{
		const char* hcd = getenv("HS_CUDA_DEVICES");
#ifdef __APPLE__
		return { -1 };
#else
		const char* cvd = getenv("CUDA_VISIBLE_DEVICES");
		int slurm_localID = getIntFromEnv("SLURM_LOCALID", -1);
		int ompi_locad_rank = getIntFromEnv("OMPI_COMM_WORLD_LOCAL_RANK", -1);
		int numGPUs;
		cudaGetDeviceCount(&numGPUs);
		std::cout << "#hs(" << world.rank << "): selecting GPUs ... " << std::endl;
		if (fromCL.forceSingleGPU) {
			std::cout << "#hs(" << world.rank << "): user requested single GPU per rank ... " << std::endl;
			if (slurm_localID >= 0) {
				int gpuID = slurm_localID % numGPUs;
				std::cout << "#hs(" << world.rank << "): "
					<< "SLURM_LOCALID=" << slurm_localID
					<< " (mod numGPUs=" << numGPUs << ")"
					<< " -> { " << gpuID << " }"
					<< " ... (that's physical GPU " << getPhysicalString(gpuID) << ")"
					<< std::endl;
				return { gpuID };
			}
			else if (hcd) {
				std::cout << "#hs(" << world.rank << "): parsing from user-supplied list " << hcd << std::endl;
				std::vector<int> gpuIDs = parseCommaSeparatedListOfInts(hcd);
				PRINT(gpuIDs.size());
				return gpuIDs;
			}
			else
				std::cout << "#hs(" << world.rank << "): "
				<< "SLURM_LOCALID=<not set> ... not using slurm(?)" << std::endl;

			int gpuID = localRank % numGPUs;
			std::cout << "#hs(" << world.rank << "): "
				<< " setting from self-determined localrank "
				<< localRank << "/" << localSize
				<< " (mod numGPUs=" << numGPUs << ")"
				<< " -> { " << gpuID << " }"
				<< " ... (that's physical GPU " << getPhysicalString(gpuID) << ")"
				<< std::endl;
			return { gpuID };
		}
		else {
			std::cout << "#hs(" << world.rank << "): user requested *multiple* GPUs per rank ... " << std::endl;
			if (cvd && slurm_localID) {
				std::cout << "#hs(" << world.rank << "): *both* SLURM_LOCALID *and* CUDA_VISIBLE_DVIES are set ... I assume slurm has pre-selected the GPUs to use, and stored it in CUDA_VISIBLE_DEVICES -> using all GPUs reported by CUDA " << std::endl;
				std::vector<int> gpuIDs;
				for (int i = 0; i < numGPUs; i++)
					gpuIDs.push_back(i);
				return gpuIDs;
			}
			std::cout << "#hs(" << world.rank << "): assume we see the actual physical devices... "
				<< "distribute these " << numGPUs << " GPUs over " << localSize
				<< " local processes..." << std::endl;
			std::vector<int> gpuIDs;
			int gpusPerRank = std::max(1, numGPUs / localSize);
			PRINT(gpusPerRank);
			assert(gpusPerRank > 0);
			for (int i = 0; i < gpusPerRank; i++)
				gpuIDs.push_back((localRank + i * localSize) % numGPUs);
			assert(!gpuIDs.empty());
			PRINT(gpuIDs.size());
			return gpuIDs;
		}
#endif
	}
}

using namespace hs;

renderengine_data g_renderengine_data_rcv;
renderengine_data g_renderengine_data;

double fps_previous_time = 0;
int fps_frame_count = 0;

void mul_vec(mini::common::vec3f& r, const float* mat, const mini::common::vec3f& vec)
{
	const float x = vec[0];
	const float y = vec[1];

	r[0] = x * mat[0 + 4 * 0] + y * mat[1 + 4 * 0] + mat[2 + 4 * 0] * vec[2];// +mat[3 + 4 * 0];
	r[1] = x * mat[0 + 4 * 1] + y * mat[1 + 4 * 1] + mat[2 + 4 * 1] * vec[2];// +mat[3 + 4 * 1];
	r[2] = x * mat[0 + 4 * 2] + y * mat[1 + 4 * 2] + mat[2 + 4 * 2] * vec[2];// +mat[3 + 4 * 2];
}

void mul_point(mini::common::vec3f& r, const float* mat, const mini::common::vec3f& vec)
{
	const float x = vec[0];
	const float y = vec[1];

	r[0] = x * mat[0 + 4 * 0] + y * mat[1 + 4 * 0] + mat[2 + 4 * 0] * vec[2] + mat[3 + 4 * 0];
	r[1] = x * mat[0 + 4 * 1] + y * mat[1 + 4 * 1] + mat[2 + 4 * 1] * vec[2] + mat[3 + 4 * 1];
	r[2] = x * mat[0 + 4 * 2] + y * mat[1 + 4 * 2] + mat[2 + 4 * 2] * vec[2] + mat[3 + 4 * 2];
}

void display_fps(int samples, double render_time, double render_time_accu, int spp_one_step)
{
	double current_time = getCurrentTime();
	fps_frame_count++;

	if (current_time - fps_previous_time >= 3.0) {

		printf("FPS: %.2f, samples: %d, render_time: %.2f, render_time_fps: %.2f, render_time_accu: %.2f, spp_one_step: %d\n", (double)fps_frame_count / (current_time - fps_previous_time), samples, render_time, 1.0 / render_time, render_time_accu, spp_one_step);
		fps_frame_count = 0;
		fps_previous_time = getCurrentTime();
	}
}

int main(int ac, char** av)
{
	/*! init ALL gpus - let's do that right away, so gpus are
		initailized before mpi even gets to run */
	hs::initAllGPUs();

	hs::mpi::init(ac, av);
#if HS_FAKE_MPI
	hs::mpi::Comm world;

#ifdef WITH_GADGET_IO_LIB
	MPI_Init(&ac, &av);
#endif

#else
	hs::mpi::Comm world(MPI_COMM_WORLD);
#endif

	world.barrier();
	if (world.rank == 0) {
		std::cout << "#hv: hsviewer starting up" << std::endl; fflush(0);
	}

	// Print GPU devices
	int device_count;
	cudaGetDeviceCount(&device_count);
	std::string dev_properties = " (";

	for (int i = 0; i < device_count; ++i) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		dev_properties += std::to_string(deviceProp.pciBusID) + std::string(":") + std::to_string(deviceProp.pciDeviceID) + std::string(",");
	}
	dev_properties += ")";

	std::cout << "#hv: rank: " << world.rank << ", GPU devices: " << device_count << dev_properties << std::endl; fflush(0);

	world.barrier();

	bool hanari = true;
	DynamicDataLoader loader(world);
	for (int i = 1; i < ac; i++) {
		const std::string arg = av[i];
		if (arg[0] != '-') {
			loader.addContent(arg);
		}
		else if (arg == "--no-bg") {
			fromCL.useBackground = false;
		}
		else if (arg == "-bg") {
			fromCL.useBackground = true;
		}
		else if (arg == "-dp" || arg == "--data-parallel") {
			fromCL.dpMode = DPMODE_DATA_PARALLEL;
		}
		else if (arg == "-sg" || arg == "--single-gpu") {
			fromCL.forceSingleGPU = true;
		}
		else if (arg == "-dp1" || arg == "-dpsg" || arg == "--data-parallel-single-gpu") {
			fromCL.dpMode = DPMODE_DATA_PARALLEL;
			fromCL.forceSingleGPU = true;
		}
		else if (arg == "-dr" || arg == "--data-replicated") {
			fromCL.dpMode = DPMODE_DATA_REPLICATED;
		}
		else if (arg == "-cm" || arg == "--color-map") {
			fromCL.cmID = std::stoi(av[++i]);
		}
		else if (arg == "-env" || arg == "--env-map") {
			fromCL.envMapFileName = av[++i];
			loader.sharedLights.envMap = fromCL.envMapFileName;
		}
		else if (arg == "--num-frames") {
			fromCL.numFramesAccum = std::stoi(av[++i]);
		}
		else if (arg == "--ambient") {
			fromCL.ambientRadiance = std::stof(av[++i]);
		}
		else if (arg == "-spp" || arg == "-ppp" || arg == "--paths-per-pixel") {
			fromCL.spp = std::stoi(av[++i]);
		}
		else if (arg == "-mum" || arg == "--merge-unstructured-meshes" || arg == "--merge-umeshes") {
			fromCL.mergeUnstructuredMeshes = true;
		}
		else if (arg == "--no-mum") {
			fromCL.mergeUnstructuredMeshes = false;
		}
		else if (arg == "--default-radius") {
			loader.defaultRadius = std::stof(av[++i]);
		}
		else if (arg == "--measure") {
			fromCL.measure = true;
		}
		else if (arg == "-o") {
			fromCL.outFileName = av[++i];
		}
		else if (arg == "--dir-light") {
			mini::DirLight light;
			light.direction = get3f(av, i);
			light.radiance = get3f(av, i);
			loader.sharedLights.directional.push_back(light);
		}
		else if (arg == "--camera-pdu") {
			fromCL.camera.vp = get3f(av, i);
			fromCL.camera.vi = fromCL.camera.vp + get3f(av, i);
			fromCL.camera.vu = get3f(av, i);
		}
		else if (arg == "--camera") {
			fromCL.camera.vp = get3f(av, i);
			fromCL.camera.vi = get3f(av, i);
			fromCL.camera.vu = get3f(av, i);
		}
		else if (arg == "-fovy") {
			fromCL.camera.fovy = std::stof(av[++i]);
		}
		else if (arg == "-xf") {
			fromCL.xfFileName = av[++i];
		}
		else if (arg == "-res" || arg == "-os" || arg == "--output-size") {
			fromCL.fbSize.x = std::stoi(av[++i]);
			fromCL.fbSize.y = std::stoi(av[++i]);
		}
		else if (arg == "-ndg") {
			fromCL.ndg = std::stoi(av[++i]);
			fromCL.dpMode
				= (fromCL.ndg == 1)
				? DPMODE_DATA_REPLICATED
				: DPMODE_DATA_PARALLEL;
		}
		else if (arg == "-dpr") {
			fromCL.dpr = std::stoi(av[++i]);
		}
		else if (arg == "-nhn" || arg == "--no-head-node") {
			fromCL.createHeadNode = false;
		}
		else if (arg == "-hn" || arg == "-chn" ||
			arg == "--head-node" || arg == "--create-head-node") {
			fromCL.createHeadNode = true;
		}
		else if (arg == "-h" || arg == "--help") {
			usage();
		}
		else if (arg == "-anari" || arg == "--hanari") {
			hanari = true;
		}
		else if (arg == "-native" || arg == "--native") {
			hanari = false;
		}
		else if (arg == "-port") {
			fromCL.port = std::stoi(av[++i]);
		}
		else {
			usage("unknown cmd-line argument '" + arg + "'");
		}
	}

	int localRank = 0, localSize = 1;
#if HS_MPI
	determineLocalProcessID(world, localRank, localSize);
#endif
	std::vector<int> gpuIDs;
	for (int i = 0; i < world.size; i++) {
		world.barrier();
		if (i != world.rank) continue;
		gpuIDs = selectGPUs(world, localRank, localSize);
	}
	assert(!gpuIDs.empty());
	world.barrier();

	const bool isHeadNode = fromCL.createHeadNode && (world.rank == 0);
	hs::mpi::Comm workers = world.split(!isHeadNode);

	if (world.size > 1 && fromCL.dpMode == DPMODE_NOT_SPECIFIED)
		throw std::runtime_error("you're running haystack in MPI mode, and with more than one rank, but didn't specify num data groups (-ndg <n>), or whether you want to run data parallel (-dp|--data-parallel) or data replicated (-dr|--data-replicated). Just to make sure we're not actually running the wrong mode I'll hereby bail out ...");

	if (fromCL.ndg == 0) {
		if (fromCL.dpMode == DPMODE_DATA_PARALLEL && world.size > 1)
			// we _are_ run in mpi mode with more than one rank, and in data
			// _parallel_mode. if not otherwise specified, use one data
			// group per rank
			fromCL.ndg = world.size;
		else
			fromCL.ndg = 1;
	}

	TcpConnection blenderClientTcp;
	//blenderClientTcp.init_sockets_cam(fromCL.server.c_str(), fromCL.port_cam, fromCL.port_data);
	blenderClientTcp.init_sockets_data("localhost", fromCL.port);

	int numDataGroupsGlobally = fromCL.ndg;
	int dataPerRank = fromCL.dpr;
	LocalModel thisRankData;
	thisRankData.colorMapIndex = fromCL.cmID;
	if (!isHeadNode) {
		loader.loadData(thisRankData, numDataGroupsGlobally, dataPerRank, verbose());
	}
	if (fromCL.mergeUnstructuredMeshes) {
		std::cout << "merging potentially separate unstructured meshes into single mesh" << std::endl;
		thisRankData.mergeUnstructuredMeshes();
		std::cout << "done mergine umeshes..." << std::endl;
	}

	int numDataGroupsLocally = thisRankData.size();
	world.barrier();
	HayMaker* hayMaker
		= hanari
		? HayMaker::createAnariImplementation(world,
			/* the workers */workers,
			fromCL.spp,
			fromCL.ambientRadiance,
			fromCL.useBackground,
			thisRankData,
			gpuIDs,
			verbose())
		: HayMaker::createBarneyImplementation(world,
			/* the workers */workers,
			fromCL.spp,
			fromCL.ambientRadiance,
			fromCL.useBackground,
			thisRankData,
			gpuIDs,
			verbose());
	// #if HANARI
	//     hayMaker = new HayMakerT<AnariBackend>(world,
	//                                            /* the workers */workers,
	//                                            thisRankData,
	//                                            verbose());
	// #else
	// #endif
	//   } else 
	//     hayMaker = new HayMakerT<BarneyBackend>(world,
	//                                             /* the workers */workers,
	//                                             thisRankData,
	//                                             verbose());

	world.barrier();
	const BoundsData worldBounds = hayMaker->getWorldBounds();
	bool modelHasVolumeData = !worldBounds.scalars.empty();

	if (world.rank == 0)
		std::cout << MINI_TERMINAL_CYAN
		<< "#hs: world bounds is " << worldBounds
		<< MINI_TERMINAL_DEFAULT << std::endl;

	if (fromCL.camera.vp == fromCL.camera.vi) {
		fromCL.camera.vp
			= worldBounds.spatial.center()
			+ mini::common::vec3f(-.3f, .7f, +1.f) * worldBounds.spatial.span();
		fromCL.camera.vi = worldBounds.spatial.center();
	}

	world.barrier();
	if (world.rank == 0)
		std::cout << MINI_TERMINAL_CYAN
		<< "#hs: creating barney context"
		<< MINI_TERMINAL_DEFAULT << std::endl;
	// hayMaker->createBarney();
	world.barrier();
	if (world.rank == 0)
		std::cout << MINI_TERMINAL_CYAN
		<< "#hs: building barney data groups"
		<< MINI_TERMINAL_DEFAULT << std::endl;
	if (!isHeadNode)
		hayMaker->buildSlots();

	world.barrier();

	Renderer* renderer = nullptr;
	if (world.size == 1)
		// no MPI, render direcftly
		renderer = hayMaker;
	else if (world.rank == 0)
		// we're in MPI mode, _and_ the rank that runs the viewer
		renderer = new MPIRenderer(world, hayMaker);
	else {
		// we're in MPI mode, but one of the passive workers (ie NOT running the viewer)
		MPIRenderer::runWorker(world, hayMaker);
		world.barrier();
		hs::mpi::finalize();

		exit(0);
	}

	////////////////////////////////////////////////////
	fromCL.camera.vp.x = 0;
	fromCL.camera.vp.y = 0;
	fromCL.camera.vp.z = 0;

	fromCL.camera.vi.x = 0;
	fromCL.camera.vi.y = 0;
	fromCL.camera.vi.z = -1;

	fromCL.camera.vu.x = 0;
	fromCL.camera.vu.y = 1;
	fromCL.camera.vu.z = 0;

	double render_time = 0;
	double render_time_accu = 0;
	int spp_one_step = 0;

	void* fbPointer = NULL;

	std::vector<char> pixels_buf_empty;
	HsDataRender hsDataRender, hsDataRenderRcv;

	TransferFunction xf;
	xf.colorMap.resize(sizeof(hsDataRender.colorMap) / sizeof(vec4f));
	xf.domain = range1f(0.f, 1.0f);

	int total_samples = 0;
	hs::Camera camera;

	double t2 = getCurrentTime();

	/////////

	HsDataState hsDataState;
	memset(&hsDataState, 0, sizeof(hsDataState));
	hsDataState.world_bounds_spatial_lower[0] = worldBounds.spatial.lower[0];
	hsDataState.world_bounds_spatial_lower[1] = worldBounds.spatial.lower[1];
	hsDataState.world_bounds_spatial_lower[2] = worldBounds.spatial.lower[2];
	hsDataState.world_bounds_spatial_upper[0] = worldBounds.spatial.upper[0];
	hsDataState.world_bounds_spatial_upper[1] = worldBounds.spatial.upper[1];
	hsDataState.world_bounds_spatial_upper[2] = worldBounds.spatial.upper[2];

	hsDataState.scalars_range[0] = worldBounds.scalars.lo;
	hsDataState.scalars_range[1] = worldBounds.scalars.hi;

	/////////

	while (true) {
		blenderClientTcp.recv_data_data((char*)&g_renderengine_data_rcv, sizeof(renderengine_data));
		if (blenderClientTcp.is_error()) {
			throw std::runtime_error("TCP Error!");
		}

		if (g_renderengine_data_rcv.reset) {
			// if (renderer != NULL) {
			// 	delete renderer;
			// 	renderer = NULL;
			// }

			blenderClientTcp.client_close();
			blenderClientTcp.server_close();

			break;
		}

		blenderClientTcp.recv_data_data((char*)&hsDataRenderRcv, sizeof(HsDataRender));
		if (blenderClientTcp.is_error()) {
			throw std::runtime_error("TCP Error!");
		}

		if (pixels_buf_empty.size() != sizeof(uint32_t) * g_renderengine_data_rcv.width * g_renderengine_data_rcv.height) {
			pixels_buf_empty.resize(sizeof(uint32_t) * g_renderengine_data_rcv.width * g_renderengine_data_rcv.height);
		}

		try {

			// cam_change
			if (/*renderer == NULL || */ memcmp(&g_renderengine_data, &g_renderengine_data_rcv, sizeof(renderengine_data))) {
				memcpy(&g_renderengine_data, &g_renderengine_data_rcv, sizeof(renderengine_data));

				//Camera cam = renderer->getCamera();
				renderer->resetAccumulation();
				total_samples = 0;
				render_time = 0;
				render_time_accu = 0;
				//renderer->config.camera.dirty = true;

				if (g_renderengine_data_rcv.reset || fromCL.fbSize.x != g_renderengine_data_rcv.width || fromCL.fbSize.y != g_renderengine_data_rcv.height) {
					fromCL.fbSize.x = g_renderengine_data_rcv.width;
					fromCL.fbSize.y = g_renderengine_data_rcv.height;

					cudaSetDevice(0);

					if (fbPointer)
						cudaFree(fbPointer);

					if (!hanari)
						cudaMalloc(&fbPointer, fromCL.fbSize.x * fromCL.fbSize.y * sizeof(uint32_t));
					else
						cudaMallocManaged(&fbPointer, fromCL.fbSize.x * fromCL.fbSize.y * sizeof(uint32_t));

					renderer->resize(fromCL.fbSize, (uint32_t*)fbPointer);
				}

				//renderer->setNumPPP(g_renderengine_data_rcv.step_samples);
				if (!hanari)
					camera.fovy = g_renderengine_data_rcv.cam.lens * 180.0 / M_PI;
				else
					camera.fovy = g_renderengine_data_rcv.cam.lens;

				//TODO
				//camera.type = (hs::CameraType)g_renderengine_data_rcv.cam.view_perspective;

				mul_point(camera.vp, g_renderengine_data_rcv.cam.transform_inverse_view_matrix, fromCL.camera.vp);
				mul_point(camera.vi, g_renderengine_data_rcv.cam.transform_inverse_view_matrix, fromCL.camera.vi);
				mul_vec(camera.vu, g_renderengine_data_rcv.cam.transform_inverse_view_matrix, fromCL.camera.vu);

				//camera.vp = fromCL.camera.vp;
				//camera.vu = fromCL.camera.vu;
				//camera.vi = fromCL.camera.vi;
				//camera.fovy = fromCL.camera.fovy;
				renderer->setCamera(camera);
			}

			// if (renderer == NULL) {
			// 	continue;
			// }

			if (memcmp(&hsDataRender, &hsDataRenderRcv, sizeof(HsDataRender))) {
				memcpy(&hsDataRender, &hsDataRenderRcv, sizeof(HsDataRender));

				memcpy(xf.colorMap.data(), hsDataRender.colorMap, sizeof(vec4f) * xf.colorMap.size());
				xf.domain = range1f(hsDataRender.domain[0], hsDataRender.domain[1]);
				xf.baseDensity = hsDataRender.baseDensity;

				//renderer->setColorMap(haystack_data);
				renderer->setTransferFunction(xf);
				renderer->resetAccumulation();
				total_samples = 0;
			}

			// spp_one_step = renderer->getTotalSamples();
			// double start_render_time = getCurrentTime();				
	  /////////////////////////////////////////////////
			//renderer->renderFrame();
			double t0 = getCurrentTime();
			//renderer->renderFrame(fromCL.spp);
			for (int i = 0; i < fromCL.numFramesAccum; i++)
				renderer->renderFrame();
			double t1 = getCurrentTime();
			static double sum_t = 0.f;
			static double sum_w = 0.f;
			sum_t = 0.8f * sum_t + (t1 - t0);
			sum_w = 0.8f * sum_w + 1.f;
			float timePerFrame = sum_t / sum_w;
			float fps = 1.f / timePerFrame;

			if (getCurrentTime() - t2 > 2.0) {
				std::string title = "HayThere (" + prettyDouble(fps) + "fps), " + std::to_string(t0) + ", " + std::to_string(t1);
				std::cout << title << std::endl;
				t2 = getCurrentTime();
			}
			/////////////////////////////////////////////////
			total_samples++;

			cudaDeviceSynchronize();

			cudaSetDevice(0);

#ifdef WITH_CLIENT_GPUJPEG     
			blenderClientTcp.send_gpujpeg((char*)fbPointer, pixels_buf_empty.data(), fromCL.fbSize.x, fromCL.fbSize.y);
#else
			char* pixels_buf = (char*)fbPointer; //renderer->getBuffer();
			//((int*)pixels_buf)[0] = total_samples; //renderer->getTotalSamples();

			blenderClientTcp.send_data_data((char*)fbPointer, pixels_buf_empty.size());
#endif
			if (blenderClientTcp.is_error()) {
				throw std::runtime_error("TCP Error!");
			}

			hsDataState.fps = fps;
			hsDataState.samples = total_samples;
			blenderClientTcp.send_data_data((char*)&hsDataState, sizeof(hsDataState));

			if (blenderClientTcp.is_error()) {
				throw std::runtime_error("TCP Error!");
			}
		}
		catch (const std::exception& ex)
		{
			std::cerr << ex.what();
			exit(-1);
		}
	}

	////////////////////////////////////////////////////  
	renderer->terminate();
	world.barrier();
	hs::mpi::finalize();

	return 0;
}
