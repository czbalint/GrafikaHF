//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec3 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, vp.z) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

class Camera2D {
    vec2 wCenter;
    vec2 wSize;
public:
    Camera2D() : wCenter(0,0), wSize(400,400) {}

    mat4 V() { return TranslateMatrix(-wCenter); }
    mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

    mat4 Vinv() { return TranslateMatrix(wCenter); }
    mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

    void Zoom(float s) { wSize = wSize * s; }
    void Pan(vec2 t) { wCenter = wCenter + t; }

};

//Camera2D camera;
GPUProgram gpuProgram; // vertex and fragment shaders
static const int nv = 100;
const float r = 0.04f;

class Node {
    unsigned int vaoNode, vboNode;
    vec2 wPoints;
    vec3 color;
public:
    Node(vec2 pos, vec3 color) : wPoints(pos), color(color){
        vec3 vertices[nv];
        glGenVertexArrays(1, &vaoNode);
        glBindVertexArray(vaoNode);
        glGenBuffers(1, &vboNode);
        glBindBuffer(GL_ARRAY_BUFFER, vboNode);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE,3 * sizeof(float), NULL);

        for (int j = 0; j < nv; j++) {
            float fi = j * 2 * M_PI / nv;
            vec2 tmp = {cosf(fi) * r + wPoints.x, sinf(fi) * r + wPoints.y};
            float d = sqrtf(powf(tmp.x, 2) + powf(tmp.y, 2) + 0);
            vec3 p = {(tmp.x / d) * sinhf(d), (tmp.y / d) * sinhf(d), coshf(d)};
            vertices[j] = p;
        }
        glBufferData(GL_ARRAY_BUFFER, nv * sizeof(vec3), vertices, GL_STATIC_DRAW);
    }

    void Draw(){
        gpuProgram.setUniform(color, "color");
        glBindVertexArray(vaoNode);
        glDrawArrays(GL_TRIANGLE_FAN, 0, nv);
    }

};

class Graph {
    unsigned int vaoPoint, vboPoint;
    unsigned int vaoLine, vboLine;
    static const int nv = 100;

    std::vector<vec2> wPoints;
    std::vector<vec2> rEdges;
    vec3 vertices[nv];
public:
    Graph() {

        glGenVertexArrays(1, &vaoPoint);
        glBindVertexArray(vaoPoint);
        glGenBuffers(1, &vboPoint);
        glBindBuffer(GL_ARRAY_BUFFER, vboPoint);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE,3 * sizeof(float), NULL);

        glGenVertexArrays(1, &vaoLine);
        glBindVertexArray(vaoLine);
        glGenBuffers(1, &vboLine);
        glBindBuffer(GL_ARRAY_BUFFER, vboLine);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0,2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL );

        SetRandomPoints(50);
        RandomEdges();

        for (int i = 0; i < wPoints.size(); i++) {
            for (int j = 0; j < nv; j++) {
                float fi = j * 2 * M_PI / nv;
                vec2 tmp = {cosf(fi) * r + wPoints[i].x, sinf(fi) * r + wPoints[i].y};
                float d = sqrtf(powf(tmp.x, 2) + powf(tmp.y, 2) + 0);
                vec3 p = {(tmp.x / d) * sinhf(d), (tmp.y / d) * sinhf(d), coshf(d)};
                vertices[i] = p;


            }
            glBindBuffer(GL_ARRAY_BUFFER, vboPoint);


        }
    }

    ~Graph(){
        glDeleteBuffers(1, &vboPoint);
        glDeleteVertexArrays(1, &vaoPoint);
        glDeleteBuffers(1, &vboLine);
        glDeleteVertexArrays(1, &vaoLine);
    }

    void AddControlPoint(float cX, float cY){
        vec4 wVertex = vec4(cX, cY, 0, 1);// * camera.Pinv() * camera.Vinv();
        wPoints.push_back(vec2(wVertex.x , wVertex.y));
    }

    void Draw(){
        mat4 VPTransform = {1,0,0,0,
                            0,1,0,0,
                            0,0,1,0,
                            0,0,0,1};
        gpuProgram.setUniform(VPTransform, "MVP");

     /*   if (wPoints.size() > 1) {

            glBindBuffer(GL_ARRAY_BUFFER, vboLine);
            //mat4 VPTransform = camera.V() * camera.P();
            //gpuProgram.setUniform(VPTransform, "MVP");
            gpuProgram.setUniform(vec3(1, 0, 0), "color");
            for (int i = 0; i < rEdges.size(); i++){
                std::vector<vec2> tmp;
                tmp.push_back(wPoints[rEdges[i].x]);
                tmp.push_back(wPoints[rEdges[i].y]);
                glBufferData(GL_ARRAY_BUFFER, tmp.size() * sizeof(vec2), &tmp[0], GL_DYNAMIC_DRAW);
                glBindVertexArray(vaoLine);
                glDrawArrays(GL_LINE_STRIP, 0, tmp.size());
            }

        }

        if (wPoints.size() > 0) {

            glBindBuffer(GL_ARRAY_BUFFER, vboPoint);

            for (int i = 0; i < wPoints.size(); i++) {
//                float d = sqrt((wPoints[i].x * wPoints[i].x) + (wPoints[i].y * wPoints[i].y) + 0);
//                vec4 p = {(wPoints[i].x / d) * sinh(d), (wPoints[i].y / d) * sinh(d), 0/d * sinh(d), cosh(d)};
//                mat4 MVPtransf = {r, 0, 0, 0,    // MVP matrix,
//                                  0, r, 0, 0,    // row-major!
//                                  0, 0, 0, 0,
//                                  p.x, p.y, p.z, p.w};
//
//                mat4 VPTransform = MVPtransf;// * camera.V() * camera.P();
//                gpuProgram.setUniform(VPTransform, "MVP");


                glBufferData(GL_ARRAY_BUFFER, nv * sizeof(vec3), &vertices[0], GL_STATIC_DRAW);
                gpuProgram.setUniform(vec3(0, 1, 0), "color");
                glBindVertexArray(vaoPoint);
                glDrawArrays(GL_TRIANGLE_FAN, 0, nv);
//                MVPtransf[0][0] = r/2;
//                MVPtransf[1][1] = r/2;
//                VPTransform = MVPtransf;// * camera.V() * camera.P();
//                gpuProgram.setUniform(VPTransform, "MVP");
//                gpuProgram.setUniform(vec3(RandomNumber(0,1), RandomNumber(0,1), RandomNumber(0,1)), "color");
//                glBindVertexArray(vaoPoint);
//                glDrawArrays(GL_TRIANGLE_FAN, 0, nv);
            }
        }*/


    }

    void SetRandomPoints(int n){
        wPoints.clear();
        for (int i = 0; i < n ; i++) {
            wPoints.push_back(vec2(RandomNumber(-1,1),RandomNumber(-1,1)));
        }
    }

    float RandomNumber(float Min, float Max) {
        return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
    }

    void RandomEdges() {
        int maxEdge = 50*49/2;
        while ((float)rEdges.size() / (float)maxEdge < 0.05f) {
            int e1 = rand() % 50;
            int e2 = rand() % 50;
            if (e1 == e2){
                continue;
            } else {
                bool has = false;
                for (auto Iterator = rEdges.begin(); Iterator != rEdges.end(); Iterator++){
                    vec2 tmp = *Iterator;
                    if (tmp.x == e1 && tmp.y == e2 || tmp.x == e2 && tmp.y == e1){
                        has = true;
                        continue;
                    }
                }
                if (!has) {
                    rEdges.push_back(vec2(e1,e2));
                }
            }
        }
        printf("%d\n",rEdges.size());
    }
};

Graph * graph;
unsigned int vao;
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
    graph = new Graph();
    glLineWidth(2.0f);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer

	graph->Draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' '){
        graph->SetRandomPoints(50);
        graph->RandomEdges();
	    glutPostRedisplay();
    }         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	//printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system

    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
	    float cx = 2.0f * pX / windowWidth - 1;
	    float cy = 1.0f - 2.0 * pY / windowHeight;
	    graph->AddControlPoint(cx, cy);
	    glutPostRedisplay();
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
