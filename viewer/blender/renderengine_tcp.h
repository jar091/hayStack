// ======================================================================== //
// Copyright 2022-2022 Ingo Wald                                            //
// Copyright 2022-2025 IT4Innovations, VSB - Technical University of Ostrava//
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

#ifndef __RENDERENGINE_TCP_H__
#define __RENDERENGINE_TCP_H__

#include <stdlib.h>

#    ifdef _WIN32

#      include <iostream>
#      include <winsock2.h>
#      include <ws2tcpip.h>

#      pragma comment(lib, "Ws2_32.lib")
#      pragma comment(lib, "Mswsock.lib")
#      pragma comment(lib, "AdvApi32.lib")

#    else
#      include <arpa/inet.h>
#      include <netdb.h>
#      include <netinet/in.h>
#      include <netinet/tcp.h>
#      include <sys/socket.h>
#      include <unistd.h>
#    endif

#ifdef WITH_CLIENT_GPUJPEG
#  include <libgpujpeg/gpujpeg_common.h>
#  include <libgpujpeg/gpujpeg_decoder.h>
#  include <libgpujpeg/gpujpeg_encoder.h>
#endif

//#define TCP_OPTIMIZATION
//#define TCP_FLOAT
#define MAX_CONNECTIONS 100


class TcpConnection {
private:
	int g_port_offset = -1;

	int g_server_id_cam[MAX_CONNECTIONS];
	int g_client_id_cam[MAX_CONNECTIONS];

	int g_server_id_data[MAX_CONNECTIONS];
	int g_client_id_data[MAX_CONNECTIONS];

	int g_timeval_sec = 60;
	int g_connection_error = 0;

	sockaddr_in g_client_sockaddr_cam[MAX_CONNECTIONS];
	sockaddr_in g_server_sockaddr_cam[MAX_CONNECTIONS];

	sockaddr_in g_client_sockaddr_data[MAX_CONNECTIONS];
	sockaddr_in g_server_sockaddr_data[MAX_CONNECTIONS];

	bool g_is_server = true;

#ifdef WITH_CLIENT_GPUJPEG
	gpujpeg_encoder* g_encoder = NULL;
	uint8_t* g_image_compressed;

	int g_compressed_quality = -1; //0-100

	gpujpeg_decoder* g_decoder = NULL;
#endif
public:
	void write_data_kernelglobal(void* data, size_t size);
	bool read_data_kernelglobal(void* data, size_t size);
	void close_kernelglobal();

	bool is_error();

	void init_sockets_cam(const char* server = NULL, int port_cam = 0, int port_data = 0, bool is_server = true);
	void init_sockets_data(const char* server = NULL, int port = 0, bool is_server = true);

	bool client_check();
	bool server_check();

	void client_close();
	void server_close();

	void send_data_cam(char* data, size_t size, bool ack = true);
	void recv_data_cam(char* data, size_t size, bool ack = true);

	void send_data_data(char* data, size_t size, bool ack = true);
	void recv_data_data(char* data, size_t size, bool ack = true);

	void send_gpujpeg(char* dmem, char* pixels, int width, int height);
	void recv_gpujpeg(char* dmem, char* pixels, int width, int height);
	void recv_decode(char* dmem, char* pixels, int width, int height, int frame_size);

	void rgb_to_yuv_i420(
		unsigned char* destination, unsigned char* source, int tile_h, int tile_w);

	void yuv_i420_to_rgb(
		unsigned char* destination, unsigned char* source, int tile_h, int tile_w);

	void yuv_i420_to_rgb_half(
		unsigned short* destination, unsigned char* source, int tile_h, int tile_w);

	void rgb_to_half(
		unsigned short* destination, unsigned char* source, int tile_h, int tile_w);

	void set_port_offset(int offset);

private:
	int setsock_tcp_windowsize(int inSock, int inTCPWin, int inSend);
	bool init_wsa();
	void init_port();
	void close_wsa();
	bool server_create(int port,
		int& server_id,
		int& client_id,
		sockaddr_in& server_sock,
		sockaddr_in& client_sock,
		bool only_accept);

	bool client_create(const char* server_name, int port, int& client_id, sockaddr_in& client_sock);
	void close_tcp(int id);

	void send_data(char* data, size_t size);
	void recv_data(char* data, size_t size);

#ifdef WITH_CLIENT_GPUJPEG
	int gpujpeg_encode(int width,
		int height,
		uint8_t* input_image,
		uint8_t* image_compressed,
		int& image_compressed_size);

	int gpujpeg_decode(int width,
		int height,
		uint8_t* input_image,
		uint8_t* image_compressed,
		int& image_compressed_size);
#endif
};

#endif
