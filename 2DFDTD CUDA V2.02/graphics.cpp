#include "graphics.h"
// CARTESIAN
unsigned int window_width = M;
unsigned int window_height = N;
unsigned int image_width = M;
unsigned int image_height = N;
int drawMode = GL_LINE_STRIP;
int iGLUTWindowHandle = 0;          // handle to the GLUT window
size_t number_of_bytes;

GLuint pbo_destination;
struct cudaGraphicsResource *cuda_pbo_destination_resource;
GLuint cuda_result_texture;

// the following set some of the external variables declared in graphics.h to zero so that 
// they do not show up as unresolved externals during early compiles
//float dx = 0.0;
float dy = 0.0;
float domain_min_x = 0.0;
float domain_min_y = 0.0;

// The variables below were also extern in global.h, but were then defined in global.cpp
// in Demir's code.
float global_min_field = 1e9;
float global_max_field = -1e9;
unsigned int* image_data;
float* field_data;

unsigned int rgb[] = { 4286775296, 4287037440, 4287299584, 4287561728, 4287823872, 4288086016, 4288348160,
4288610304, 4288872448, 4289134592, 4289396736, 4289658880, 4289921024, 4290183168,
4290445312, 4290707456, 4290969600, 4291231744, 4291493888, 4291756032, 4292018176,
4292280320, 4292542464, 4292804608, 4293066752, 4293328896, 4293591040, 4293853184,
4294115328, 4294377472, 4294639616, 4294901760, 4294902784, 4294903808, 4294904832,
4294905856, 4294906880, 4294907904, 4294908928, 4294909952, 4294910976, 4294912000,
4294913024, 4294914048, 4294915072, 4294916096, 4294917120, 4294918144, 4294919168,
4294920192, 4294921216, 4294922240, 4294923264, 4294924288, 4294925312, 4294926336,
4294927360, 4294928384, 4294929408, 4294930432, 4294931456, 4294932480, 4294933504,
4294934528, 4294935296, 4294936320, 4294937344, 4294938368, 4294939392, 4294940416,
4294941440, 4294942464, 4294943488, 4294944512, 4294945536, 4294946560, 4294947584,
4294948608, 4294949632, 4294950656, 4294951680, 4294952704, 4294953728, 4294954752,
4294955776, 4294956800, 4294957824, 4294958848, 4294959872, 4294960896, 4294961920,
4294962944, 4294963968, 4294964992, 4294966016, 4294967040, 4294704900, 4294442760,
4294180620, 4293918480, 4293656340, 4293394200, 4293132060, 4292869920, 4292607780,
4292345640, 4292083500, 4291821360, 4291559220, 4291297080, 4291034940, 4290772800,
4290510660, 4290248520, 4289986380, 4289724240, 4289462100, 4289199960, 4288937820,
4288675680, 4288413540, 4288151400, 4287889260, 4287627120, 4287364980, 4287102840,
4286840700, 4286644096, 4286381955, 4286119815, 4285857675, 4285595535, 4285333395,
4285071255, 4284809115, 4284546975, 4284284835, 4284022695, 4283760555, 4283498415,
4283236275, 4282974135, 4282711995, 4282449855, 4282187715, 4281925575, 4281663435,
4281401295, 4281139155, 4280877015, 4280614875, 4280352735, 4280090595, 4279828455,
4279566315, 4279304175, 4279042035, 4278779895, 4278517755, 4278255615, 4278254591,
4278253567, 4278252543, 4278251519, 4278250495, 4278249471, 4278248447, 4278247423,
4278246399, 4278245375, 4278244351, 4278243327, 4278242303, 4278241279, 4278240255,
4278239231, 4278238207, 4278237183, 4278236159, 4278235135, 4278234111, 4278233087,
4278232063, 4278231039, 4278230015, 4278228991, 4278227967, 4278226943, 4278225919,
4278224895, 4278223871, 4278223103, 4278222079, 4278221055, 4278220031, 4278219007,
4278217983, 4278216959, 4278215935, 4278214911, 4278213887, 4278212863, 4278211839,
4278210815, 4278209791, 4278208767, 4278207743, 4278206719, 4278205695, 4278204671,
4278203647, 4278202623, 4278201599, 4278200575, 4278199551, 4278198527, 4278197503,
4278196479, 4278195455, 4278194431, 4278193407, 4278192383, 4278191359, 4278190335,
4278190331, 4278190327, 4278190323, 4278190319, 4278190315, 4278190311, 4278190307,
4278190303, 4278190299, 4278190295, 4278190291, 4278190287, 4278190283, 4278190279,
4278190275, 4278190271, 4278190267, 4278190263, 4278190259, 4278190255, 4278190251,
4278190247, 4278190243, 4278190239, 4278190235, 4278190231, 4278190227, 4278190223,
4278190219, 4278190215, 4278190211, 4278190208 };

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -2.0;

void setImageAndWindowSize()
{
	image_width = M;
	image_height = N;

	if (M>N)
		window_height = window_width*N / M;
	else
		window_width = window_height*M / N;
}

////////////////////////////////////////////////////////////////////////////////
void createPixelBufferObject(GLuint* pbo, struct cudaGraphicsResource **pbo_resource)
{
	// set up vertex data parameter
	unsigned int texture_size;

	texture_size = sizeof(GLubyte) * image_width * image_height * 4;
	void *data = malloc(texture_size);

	// create buffer object
	glGenBuffers(1, pbo);
	glBindBuffer(GL_ARRAY_BUFFER, *pbo);
	glBufferData(GL_ARRAY_BUFFER, texture_size, data, GL_DYNAMIC_DRAW);
	free(data);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	//cutilSafeCall(cudaGraphicsGLRegisterBuffer(pbo_resource, *pbo, cudaGraphicsMapFlagsNone));
	cudaGraphicsGLRegisterBuffer(pbo_resource, *pbo, cudaGraphicsMapFlagsNone);

	SDK_CHECK_ERROR_GL();
}

void deletePBO(GLuint* pbo)
{
	glDeleteBuffers(1, pbo);
	SDK_CHECK_ERROR_GL();
	*pbo = 0;
}

// runIterationAndDisplay image to the screen as textured quad
void displayTextureImage(GLuint texture)
{
	glBindTexture(GL_TEXTURE_2D, texture);
	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//glOrtho(domain_min_x, domain_min_x + M*dx, domain_min_y, domain_min_y + N*dy, -1.0, 1.0);
	//glOrtho(0.0f, window_width, window_height, 0.0f, 0.0f, 1.0f);
	glOrtho(0.0f, 512.0f, 512.0f, 0.0f, 0.0f, 1.0f);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, window_width, window_height);

	glBegin(GL_QUADS);
	//glTexCoord2f(0.0, 0.0); glVertex3f(domain_min_x, domain_min_y, 0.0);
	//glTexCoord2f(1.0, 0.0); glVertex3f(domain_min_x + M*dx, domain_min_y, 0.0);
	//glTexCoord2f(1.0, 1.0); glVertex3f(domain_min_x + M*dx, domain_min_y + N*dy, 0.0);
	//glTexCoord2f(0.0, 1.0); glVertex3f(domain_min_x, domain_min_y + N*dy, 0.0);
	glTexCoord2f(0.0, 0.0); glVertex3f(0.0, 0.0, 0.0);
	glTexCoord2f(1.0, 0.0); glVertex3f(512.0, 0.0, 0.0);
	glTexCoord2f(1.0, 1.0); glVertex3f(512.0, 512.0, 0.0);
	glTexCoord2f(0.0, 1.0); glVertex3f(0.0, 512.0, 0.0);

	glEnd();

	glDisable(GL_TEXTURE_2D);

	//glCallList(objects_display_list);

	SDK_CHECK_ERROR_GL();
}

void find_min_and_max_on_cpu(float* field_data)
{
	float min = 1.0e50;
	float max = -1.0e50;
	float cval;
	for (int i = 0; i<M*N; i++)
	{
		cval = field_data[i];
		if (cval<min) min = cval;
		if (cval>max) max = cval;
	}
	if (min>0.0) min = 0.0;
	if (max<0.0) max = 0.0;
	if (abs(min)>max) max = -min; else min = -max;
	if (min<global_min_field) global_min_field = min;
	if (max>global_max_field) global_max_field = max;

}
void create_image_on_cpu(unsigned int* image_data, float* field_data, float minval, float maxval)
{

	int cind;
	float F;

	for (int i = 0; i<M*N; i++)
	{
		F = field_data[i] - minval;
		cind = floor(255 * F / (maxval - minval));
		if (cind > 255) cind = 255;
		image_data[i] = rgb[cind];		// rgb is declared as extern int[] in graphics.h
	}
}


void idle()
{
	glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int x, int y)
{
	switch (key) {
	case(27) :
		exit(0);
		break;
	case 'd':
	case 'D':
		switch (drawMode) {
		case GL_POINTS: drawMode = GL_LINE_STRIP; break;
		case GL_LINE_STRIP: drawMode = GL_TRIANGLE_FAN; break;
		default: drawMode = GL_POINTS;
		} break;
	}
	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx, dy;
	dx = x - mouse_old_x;
	dy = y - mouse_old_y;

	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2;
		rotate_y += dx * 0.2;
	}
	else if (mouse_buttons & 4) {
		translate_z += dy * 0.01;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void reshape(int w, int h)
{
	window_width = w;
	window_height = h;
}

////////////////////////////////////////////////////////////////////////////////
void createTextureDestination(GLuint* cuda_result_texture, unsigned int size_x, unsigned int size_y)
{
	// create a texture
	glGenTextures(1, cuda_result_texture);
	glBindTexture(GL_TEXTURE_2D, *cuda_result_texture);

	// set basic parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
void deleteTexture(GLuint* tex)
{
	glDeleteTextures(1, tex);
	SDK_CHECK_ERROR_GL();
	*tex = 0;
}

void createVBO(mappedBuffer_t* mbuf)	//void createVBO(GLuint* vbo, unsigned int typeSize)
{
	// create buffer object
	glGenBuffers(1, &(mbuf->vbo));
	glBindBuffer(GL_ARRAY_BUFFER, mbuf->vbo);

	// initialize buffer object
	unsigned int size = M * N * mbuf->typeSize;
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

#ifdef USE_CUDA3
	cudaGraphicsGLRegisterBuffer(&(mbuf->cudaResource), mbuf->vbo,
		cudaGraphicsMapFlagsNone);
#else
	// register buffer object with CUDA
	cudaGLRegisterBufferObject(mbuf->vbo);
#endif
}

void deleteVBO(mappedBuffer_t* mbuf)	//void deleteVBO(GLuint* vbo)
{
	glBindBuffer(1, mbuf->vbo);
	glDeleteBuffers(1, &(mbuf->vbo));

#ifdef USE_CUDA3
	cudaGraphicsUnregisterResource(mbuf->cudaResource);
	mbuf->cudaResource = NULL;
	mbuf->vbo = NULL;
#else
	cudaGLUnregisterBufferObject(mbuf->vbo);
	mbuf->vbo = NULL;
#endif

}

void cleanupCudaVbo()
{
	deleteVBO(&vertexVBO);
	deleteVBO(&colorVBO);
}

void initCudaVbo()
{
	pickGPU(1);
	createVBO(&vertexVBO);
	createVBO(&colorVBO);
}

void renderCudaVbo(int drawMode)	// This is used instead of displayTextureImage (as in 2DFDTD CUDA V2.01)
{
	glBindBuffer(GL_ARRAY_BUFFER, vertexVBO.vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, colorVBO.vbo);
	glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);

	//glDrawArrays(GL_POINTS, 0, g->nCells);

	switch (drawMode) {
	case GL_LINE_STRIP:
		for (int i = 0; i < M*N; i += M)
			glDrawArrays(GL_LINE_STRIP, i, M);
		break;
	case GL_TRIANGLE_FAN: {
		static GLuint* qIndices = NULL;
		int size = 5 * (N - 1)*(M - 1);

		if (qIndices == NULL) { // allocate and assign trianglefan indicies 
			qIndices = (GLuint *)malloc(size*sizeof(GLint));
			int index = 0;
			for (int i = 1; i < N; i++) {
				for (int j = 1; j < M; j++) {
					qIndices[index++] = (i)*M + j;
					qIndices[index++] = (i)*M + j - 1;
					qIndices[index++] = (i - 1)*M + j - 1;
					qIndices[index++] = (i - 1)*M + j;
					qIndices[index++] = RestartIndex;
				}
			}
		}
		glPrimitiveRestartIndexNV(RestartIndex);
		glEnableClientState(GL_PRIMITIVE_RESTART_NV);
		glDrawElements(GL_TRIANGLE_FAN, size, GL_UNSIGNED_INT, qIndices);
		glDisableClientState(GL_PRIMITIVE_RESTART_NV);
	} break;
	default:
		glDrawArrays(GL_POINTS, 0, M * N);
		break;
	}



	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}

void CleanupVbo(int iExitCode)
{
	deleteVBO(&vertexVBO);
	deleteVBO(&colorVBO);
	cudaThreadExit();
	if (iGLUTWindowHandle)glutDestroyWindow(iGLUTWindowHandle);
	exit(iExitCode);
}


bool runFdtdWithFieldDisplayVbo(int argc, char** argv)	// This is the primary function main() calls 
{													// and it start the glutMainLoop() which calls the display() function
	//initializeGL(argc, argv);
	initGLVbo(argc, argv);
	initCudaVbo();
	glutDisplayFunc(runIterationsAndDisplayVbo);	// sets the active GLUT display function
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);							// picks a GPU, creates both vertex buffers
	//create_Grid_points_only(dptr);
	//	createColormapOnGpu();							// colormap used to map field intensity

	if (int ret = copyTMzArraysToDevice() != 0){	// copy data from CPU RAM to GPU global memory
		if (ret == 1) printf("Memory allocation error in copyTMzArraysToDevice(). \n\n Exiting.\n");
		return 0;
	}
	glutMainLoop();									// This starts the glutMainLoop 
	Cleanup(EXIT_FAILURE);
}

void runIterationsAndDisplayVbo()						// This is the glut display function.  It is called once each
{													// glutMainLoop() iteration.
	if (g->time < maxTime)
		update_all_fields_CUDA();	// was fdtdIternationsOnGpu()
	else
	{
		copyFieldSnapshotsFromDevice();
		deallocateCudaArrays();
		Cleanup(EXIT_SUCCESS);
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);
	
	// The following block maps two cudaGraphicsResources, the color and vertex vbo's
	// and gets pointers to them.
	size_t start;

	checkCudaErrors(cudaGraphicsMapResources(1, &vertexVBO.cudaResource, NULL));	// this vertex block needs to go to the create_grid_points function.  It gets called only once
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &start,	// as this buffer does not change once initialized.
		vertexVBO.cudaResource));
	checkCudaErrors(cudaGraphicsMapResources(1, &colorVBO.cudaResource, NULL));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&cptr, &start, colorVBO.cudaResource));

	createImageOnGpuVbo();		// calls min-max and create_image kernels
	
	//create_Grid_points_only(dptr, dev_ez_float);
	create_Grid_points_only(dptr, dev_ez_float);

	// unmap the GL buffer
	checkCudaErrors(cudaGraphicsUnmapResources(1, &vertexVBO.cudaResource, NULL));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &colorVBO.cudaResource, NULL));

	renderCudaVbo(drawMode);				// Render the graphics to the back buffer
	cudaThreadSynchronize();
	glutSwapBuffers();			// Swap the front and back buffers*/
	glutPostRedisplay();		// handled by the idle() callback function
}

bool initializeGLVbo(int argc, char **argv)
{
	setImageAndWindowSize();
	// Create GL context
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(window_width, window_height);
	iGLUTWindowHandle = glutCreateWindow("CUDA OpenGL FDTD");

	// initialize necessary OpenGL extensions
	glewInit();
	if (!glewIsSupported(
		"GL_VERSION_2_0 "
		"GL_ARB_pixel_buffer_object "
		"GL_EXT_framebuffer_object "
		)) {
		printf("ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return 1;
	}

	// Initialize GLUT event functions
	glutDisplayFunc(runIterationsAndDisplay);		// runIterationAndDisplay is what glutMainLoop() will keep running.
	glutKeyboardFunc(keyboard);						// So it has to contain the FDTD time iteration loop
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutIdleFunc(idle);

	SDK_CHECK_ERROR_GL();
	return 0;
}

void initGLVbo(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop Demo (adapted from NVIDIA's simpleGL");
	glutDisplayFunc(runIterationsAndDisplay);		// runIterationAndDisplay is what glutMainLoop() will keep running.
	glutKeyboardFunc(keyboard);						// So it has to contain the FDTD time iteration loop
	//glutReshapeFunc(reshape);
	//glutMouseFunc(mouse);
	glutMotionFunc(motion);
	//glutIdleFunc(idle);

	// initialize necessary OpenGL extensions
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height,
		0.5, 45.0);


	//glMatrixMode(GL_PROJECTION);
	//glLoadIdentity();
	//glOrtho(domain_min_x, domain_min_x + M*dx, domain_min_y, domain_min_y + N*dy, -1.0, 1.0);
	//glOrtho(0.0f, window_width, window_height, 0.0f, 0.0f, 1.0f);
	//glOrtho(0.0f, 512.0f, 512.0f, 0.0f, 0.0f, 1.0f);

	//glMatrixMode(GL_MODELVIEW);
	//glViewport(0, 0, window_width, window_height);


}



////////////////////////////////////////////////////////////////////////////////
void initializeGLBuffers()
{
	// create pixel buffer object
	createPixelBufferObject(&pbo_destination, &cuda_pbo_destination_resource);
	// create texture that will receive the result of CUDA
	createTextureDestination(&cuda_result_texture, image_width, image_height);
	SDK_CHECK_ERROR_GL();
}



////////////////////////////////////////////////////////////////////////////////
void Cleanup(int iExitCode)
{
	//    cutilSafeCall(cudaGraphicsUnregisterResource(cuda_pbo_destination_resource));
	cudaGraphicsUnregisterResource(cuda_pbo_destination_resource);
	deletePBO(&pbo_destination);
	deleteTexture(&cuda_result_texture);
	cudaThreadExit();
	if (iGLUTWindowHandle)glutDestroyWindow(iGLUTWindowHandle);
	exit(iExitCode);
}

bool runFdtdWithFieldDisplay(int argc, char** argv)
{
	pickGPU(1);								// Initialize CUDA context

	//if (false == initializeGL(argc, argv))	// Initialize GL context
	//	return 0;
	initializeGL(argc, argv);
	initializeGLBuffers();					// Initialize GL buffers
	createColormapOnGpu();					// colormap used to map field intensity
		
	// copy data from CPU RAM to GPU global memory
	if (int ret = copyTMzArraysToDevice() != 0)
	{
		if (ret == 1) printf("Memory allocation error in copyTMzArraysToDevice(). \n\n Exiting.\n");
		return 0;
	}

	//time(&start);

	glutMainLoop();	// GLUT loop 

	Cleanup(EXIT_FAILURE);
}

void runIterationsAndDisplay()
{
	int plotting_steps = 10;
	// run a number of FDTD iterations on GPU using CUDA
//	for (int i = 0; i< plotting_step; i++)
//	{
		if (g->time < maxTime)
			update_all_fields_CUDA();	// was fdtdIternationsOnGpu()
		else                        
		{
			copyFieldSnapshotsFromDevice();
			deallocateCudaArrays();
			//deallocateArrays();		// This is handled by delete(g) in main.cpp
			//saveSampledFieldsToFile();// not doing this right now.
			Cleanup(EXIT_SUCCESS);
		}
//	}

	// Create image of field using CUDA
	// map the GL buffer to CUDA
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_destination_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&image_data, &number_of_bytes, cuda_pbo_destination_resource));

	// execute CUDA kernel
	createImageOnGpu(image_data);
	// unmap the GL buffer
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_destination_resource, 0));

	// Create a texture from the buffer
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_destination);
	glBindTexture(GL_TEXTURE_2D, cuda_result_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	SDK_CHECK_ERROR_GL();
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// draw the image
	displayTextureImage(cuda_result_texture);
	cudaThreadSynchronize();
	// swap the front and back buffers
	glutSwapBuffers();
}

// Initialize GL
bool initializeGL(int argc, char **argv)
{
	setImageAndWindowSize();

	// Create GL context
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(window_width, window_height);
	iGLUTWindowHandle = glutCreateWindow("CUDA OpenGL FDTD");

	// initialize necessary OpenGL extensions
	glewInit();
	if (!glewIsSupported(
		"GL_VERSION_2_0 "
		"GL_ARB_pixel_buffer_object "
		"GL_EXT_framebuffer_object "
		)) {
		printf("ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return 1;
	}

	// Initialize GLUT event functions
	glutDisplayFunc(runIterationsAndDisplay);		// runIterationAndDisplay is what glutMainLoop() will keep running.
	glutKeyboardFunc(keyboard);						// So it has to contain the FDTD time iteration loop
	glutReshapeFunc(reshape);
	glutIdleFunc(idle);

	SDK_CHECK_ERROR_GL();
	return 0;
}

bool deallocateArrays() {
	//needs deallocation code
	return true;
}

bool saveSampledFieldsToFile()
{

	/*strcpy(output_file_name, "result_");
	strncat(output_file_name, input_file_name, strlen(input_file_name));

	ofstream output_file;
	output_file.open(output_file_name, ios::out | ios::binary);

	if (!output_file.is_open())
	{
		cout << "File <" << output_file_name << "> can not be opened! \n";
		return false;
	}
	else
	{
		output_file.write((char*)sampled_electric_fields_sampled_value, number_of_sampled_electric_fields*sizeof(float)*number_of_time_steps);
		output_file.write((char*)sampled_magnetic_fields_sampled_value, number_of_sampled_magnetic_fields*sizeof(float)*number_of_time_steps);
		free(sampled_electric_fields_sampled_value);
		free(sampled_magnetic_fields_sampled_value);
	}

	output_file.close();*/
	return true;
}

