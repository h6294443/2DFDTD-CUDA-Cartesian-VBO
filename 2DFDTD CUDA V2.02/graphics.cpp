// This file belongs to 2DFDTD CUDA V2.02 (Cartesian VBO CUDA)

#include "graphics.h"

unsigned int window_width = 1080;
unsigned int window_height = 1080;
unsigned int image_width; 
unsigned int image_height; 
int drawMode = GL_TRIANGLE_FAN;
int iGLUTWindowHandle = 0;          // handle to the GLUT window
float global_min_field = 1e9;		// used for finding min/max in Ez field
float global_max_field = -1e9;		// which is used to scale color to field intensity

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -2.0;

void setImageAndWindowSize(){
	image_width = g->M;
	image_height = g->N;

	if (g->M>g->N)
		window_height = window_width*g->N / g->M;
	else
		window_width = window_height*g->M / g->N;
}

void idle(){
	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
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

void mouse(int button, int state, int x, int y) {
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

void motion(int x, int y) {
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4) {
		translate_z += dy * 0.01f;
	}
	mouse_old_x = x;
	mouse_old_y = y;
}

void reshape(int w, int h) {
	window_width = w;
	window_height = h;
}

void createVBO(mappedBuffer_t* mbuf) {
	// create buffer object
	glGenBuffers(1, &(mbuf->vbo));
	glBindBuffer(GL_ARRAY_BUFFER, mbuf->vbo);

	// initialize buffer object
	unsigned int size = g->nCells * mbuf->typeSize;
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

#ifdef USE_CUDA3
	cudaGraphicsGLRegisterBuffer(&(mbuf->cudaResource), mbuf->vbo,
		cudaGraphicsMapFlagsNone);
#else // register buffer object with CUDA
	cudaGLRegisterBufferObject(mbuf->vbo);
#endif
}

void deleteVBO(mappedBuffer_t* mbuf) {
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

void cleanupCudaVbo() {
	deleteVBO(&vertexVBO);
	deleteVBO(&colorVBO);
}

void initCudaVbo() {
	pickGPU(0);
	createVBO(&vertexVBO);
	createVBO(&colorVBO);
}

void renderCudaVbo(int drawMode) {
	glBindBuffer(GL_ARRAY_BUFFER, vertexVBO.vbo);	// bind the vertex buffer
	glVertexPointer(4, GL_FLOAT, 0, 0);				// tell OpenGL it's a vertex buffer
	glEnableClientState(GL_VERTEX_ARRAY);			// turn it on

	glBindBuffer(GL_ARRAY_BUFFER, colorVBO.vbo);	// bind the color buffer (Ez field intensity)
	glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);		// tell OpenGL it's a color buffer
	glEnableClientState(GL_COLOR_ARRAY);			// turn it on
	
	switch (drawMode) {
	case GL_LINE_STRIP:
		for (int i = 0; i < g->nCells; i += g->M)
			glDrawArrays(GL_LINE_STRIP, i, g->M);
		break;
	case GL_TRIANGLE_FAN: {
		static GLuint* qIndices = NULL;
		int size = 5 * (g->N - 1)*(g->M - 1);

		if (qIndices == NULL) { // allocate and assign trianglefan indicies 
			qIndices = (GLuint *)malloc(size*sizeof(GLint));
			int index = 0;
			for (int i = 1; i < g->N; i++) {
				for (int j = 1; j < g->M; j++) {
					qIndices[index++] = (i)*g->M + j;
					qIndices[index++] = (i)*g->M + j - 1;
					qIndices[index++] = (i - 1)*g->M + j - 1;
					qIndices[index++] = (i - 1)*g->M + j;
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
		glDrawArrays(GL_POINTS, 0, g->nCells);
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
	initGLVbo(argc, argv);
	initCudaVbo();
	glutDisplayFunc(runIterationsAndDisplayVbo);	// sets the active GLUT display function
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);							// picks a GPU, creates both vertex buffers
	//create_Grid_points_only(dptr);
	
	if (int ret = copyTMzArraysToDevice() != 0){	// copy data from CPU RAM to GPU global memory
		if (ret == 1) printf("Memory allocation error in copyTMzArraysToDevice(). \n\n Exiting.\n");
		return 0;
	}
	glutMainLoop();									// This starts the glutMainLoop 
	CleanupVbo(EXIT_FAILURE);
	return 1;
}

void runIterationsAndDisplayVbo()						// This is the glut display function.  It is called once each
{													// glutMainLoop() iteration.
	if (g->time < g->maxTime)
		update_all_fields_CUDA();	// was fdtdIternationsOnGpu()
	else
	{
		copyFieldSnapshotsFromDevice();
		deallocateCudaArrays();
		CleanupVbo(EXIT_SUCCESS);
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
	create_Grid_points_only(dptr, dev_ez_float);

	// unmap the GL buffer
	checkCudaErrors(cudaGraphicsUnmapResources(1, &vertexVBO.cudaResource, NULL));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &colorVBO.cudaResource, NULL));

	renderCudaVbo(drawMode);				// Render the graphics to the back buffer
	cudaThreadSynchronize();
	glutSwapBuffers();			// Swap the front and back buffers*/
	glutPostRedisplay();		// handled by the idle() callback function
}

void initGLVbo(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("2D-FDTD Simulation with CUDA and OpenGL (adapted from NVIDIA's simpleGL");
	glutDisplayFunc(runIterationsAndDisplayVbo);		// runIterationAndDisplay is what glutMainLoop() will keep running.
	glutKeyboardFunc(keyboard);						// So it has to contain the FDTD time iteration loop
	glutMotionFunc(motion);
	
	// initialize necessary OpenGL extensions
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);
	glViewport(0, 0, window_width, window_height);
	glMatrixMode(GL_MODELVIEW);						// set view matrix
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);					// projection
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height,
		0.5, 45.0);
}

bool saveSampledFieldsToFile(){

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

