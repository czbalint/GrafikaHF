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
    uniform vec3 m1;
    uniform vec3 m2;
    uniform float d;
	layout(location = 0) in vec3 vp;	// Varying input: vp = vertex position is expected in attrib array 0
    layout(location = 1) in vec2 uv;

    out vec2 nodeTextCoord;

    float Lorentz(vec3 p1, vec3 p2){
        return dot(vec2(p1.x,p1.y), vec2(p2.x, p2.y)) - p1.z * p2.z;
    }

    float distance(vec3 p1, vec3 p2){
        return acosh(-Lorentz(p1,p2));
    }

    vec3 mirror(vec3 p1, vec3 n){
        float d = distance(p1, n);
        vec3 v = (n - p1 * cosh(d));
        return p1 * cosh(2.0f * d) + v * 2.0f * cosh(d);
    }

	void main() {
        nodeTextCoord = uv;
        vec3 mirorVP;
        if (sinh(d) == 0){
            mirorVP = vp;
        } else {
            mirorVP = mirror(mirror(vp, m1),m2);
        }

		gl_Position = vec4(mirorVP.x, mirorVP.y, 0, mirorVP.z);		// transform vp from modeling space to normalized device space
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

const char * const fragmantNode = R"(
    #version 330			// Shader 3.3
    precision highp float;	// normal floats, makes no difference on desktop computers

    uniform sampler2D textureUnit;
    in vec2 nodeTextCoord;

    out vec4 outColor;		// computed color of the current pixel

    void main() {
        outColor = texture(textureUnit, nodeTextCoord);	// computed color is the color of the primitive
    }
)";

GPUProgram gpuProgramLine; // vertex and fragment shaders
GPUProgram gpuProgramNode;

float dis = 0;
bool pressed = false;
static const int nv = 100;
vec3 m1, m2;
vec3 prob;

float Lorentz(vec3 p1, vec3 p2){
    return dot(vec2(p1.x,p1.y), vec2(p2.x, p2.y)) - p1.z * p2.z;
}

float distance(vec3 p1, vec3 p2){
    return acoshf(-Lorentz(p1,p2));
}

vec3 mirror(vec3 p1, vec3 n){
    float d = distance(p1, n);
    vec3 v = (n - p1 * coshf(d));
    return p1 * cosh(2.0f * d) + v * 2.0f * coshf(d);
}

class Node {
    unsigned int vaoNode, vboNode;
    unsigned int vboTexture;
    Texture * texture;
    vec3 hPoint;
    vec3 color1;
    vec3 color2;
    vec3 vertices[nv];
    int cluster;
    float minDis;
public:
    Node(vec2 pos) : cluster(-1), minDis(__DBL_MAX__), hPoint(TransformToHyperbola(pos)){
        vec2 uvVerticices[nv];
        glGenVertexArrays(1, &vaoNode);
        glBindVertexArray(vaoNode);
        glGenBuffers(1, &vboNode);
        glBindBuffer(GL_ARRAY_BUFFER, vboNode);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE,3 * sizeof(float), NULL);

        glGenBuffers(1, &vboTexture);
        glBindBuffer(GL_ARRAY_BUFFER, vboTexture);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1,2,GL_FLOAT, GL_FALSE, 0, NULL);

        for (int j = 0; j < nv; j++) {
            float fi = j * 2 * M_PI / nv;
            vertices[j] = TransformToHyperbola({cosf(fi) * 0.05f, sinf(fi) * 0.05f});
            uvVerticices[j] = vec2(0.5f + 1.0f * cosf(fi), 0.5f + 1.0f * sinf(fi));
        }

        vec3 nPos = GetwPonts();
        vec3 origo = {0,0,1};
        float dis = distance(origo, nPos);
        if (sinhf(dis) != 0){
            vec3 vPont = (nPos - origo * coshf(dis)) / sinhf(dis);
            vec3 m1 = origo * coshf(dis / 4.0f) + vPont * sinhf(dis / 4.0f);
            vec3 m2 = origo * coshf((3.0f * dis) / 4.0f) + vPont * sinhf((dis * 3.0f) / 4.0f);

            for (int i = 0; i < nv; i++){
                vertices[i] = mirror(mirror(vertices[i], m1), m2);
            }
        }

        RandomColor();
        genTex(color1,color2);

        glBindBuffer(GL_ARRAY_BUFFER, vboNode);
        glBufferData(GL_ARRAY_BUFFER, nv * sizeof(vec3), vertices, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, vboTexture);
        glBufferData(GL_ARRAY_BUFFER, nv * sizeof(vec2), uvVerticices, GL_DYNAMIC_DRAW);

    }

    vec3 GetwPonts(){
        return hPoint;
    }

    vec3 TransformToHyperbola(vec2 pont){
        float d = sqrtf(powf(pont.x, 2) + powf(pont.y, 2) + 0);
        vec3 p = {(pont.x / d) * sinhf(d), (pont.y / d) * sinhf(d), coshf(d)};
        return p;
    }

    void TransformOnHyperbola(){

        for (int i = 0; i < nv; i++){
            vertices[i] = mirror(mirror(vertices[i], m1), m2);
        }
        hPoint = mirror(mirror(hPoint, m1), m2);
            printf("aaa\n"),
        glBindVertexArray(vaoNode);
        glBindBuffer(GL_ARRAY_BUFFER, vboNode);
        glBufferData(GL_ARRAY_BUFFER, nv * sizeof(vec3), vertices, GL_DYNAMIC_DRAW);
    }

    void TransformOnHyperbola(vec3 m1m, vec3 m2m){

        for (int i = 0; i < nv; i++){
            vertices[i] = mirror(mirror(vertices[i], m1m), m2m);
        }
        hPoint = mirror(mirror(hPoint, m1m), m2m);
        //printf("m1 = %f\n",m1.x),
        glBindVertexArray(vaoNode);
        glBindBuffer(GL_ARRAY_BUFFER, vboNode);
        glBufferData(GL_ARRAY_BUFFER, nv * sizeof(vec3), vertices, GL_DYNAMIC_DRAW);
    }

    float RandomNumber(float Min, float Max) {
        return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
    }

    void RandomColor(){
        color1 = {RandomNumber(0,1), RandomNumber(0,1), RandomNumber(0,1)};
        color2 = {RandomNumber(0,1), RandomNumber(0,1), RandomNumber(0,1)};
        if (length(color1 - color2) < 0.3f) RandomColor();
    }

    void genTex(vec3 color1, vec3 color2){
        std::vector<vec4> textureColor;
        for (int i = 0; i < 50; i++)
            for (int j = 0; j < 50; j++){
                float x = (float)i / 50.0f;
                float y = (float)j / 50.0f;
                if (x <= 0.5 and y <= 0.5 or x >= 0.5 and y <= 0.5)
                    textureColor.emplace_back(color1.x,color1.y,color1.z, 1);

                if (x >= 0.5 and y >= 0.5 or x <= 0.5 and y <= 0.5)
                    textureColor.push_back(vec4(color2.x,color2.y,color2.z, 1));



//                if (powf(x - 0.5f,2) + powf(y - 0.5, 2) <= 0.1f)
//                    textureColor.push_back(vec4(color1.x,color1.y,color1.z, 1));
//                else
//                    textureColor.push_back(vec4(color2.x,color2.y,color2.z, 1));
            }
        texture = new Texture(50,50,textureColor);
    }

    int getCluster(){ return cluster; }
    void setCluster(int cluster) { this->cluster = cluster; }

    float getMinDis(){ return minDis; }
    void setMinDis(float dist) { this->minDis = dist; }

    void Draw(){

        gpuProgramNode.Use();
        gpuProgramNode.setUniform(*texture, "textureUnit");

        glBindVertexArray(vaoNode);
        glDrawArrays(GL_TRIANGLE_FAN, 0, nv);

    }
};

class Graph {
    unsigned int vaoLine, vboLine;
    std::vector<Node> wPoints;
    std::vector<vec2> rEdges;
public:
    Graph() {
        glGenVertexArrays(1, &vaoLine);
        glBindVertexArray(vaoLine);
        glGenBuffers(1, &vboLine);
        glBindBuffer(GL_ARRAY_BUFFER, vboLine);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), NULL );
        SetRandomPoints(50);
        RandomEdges();
    }

    void Draw(){

        gpuProgramLine.Use();
        gpuProgramLine.setUniform(m1, "m1");
        gpuProgramLine.setUniform(m2, "m2");
        gpuProgramLine.setUniform(dis, "d");

        glBindBuffer(GL_ARRAY_BUFFER, vboLine);
        gpuProgramLine.setUniform(vec3(1, 0, 0), "color");
        for (int i = 0; i < rEdges.size(); i++) {
            std::vector<vec3> tmp;
            tmp.push_back(wPoints[rEdges[i].x].GetwPonts());
            tmp.push_back(wPoints[rEdges[i].y].GetwPonts());
            glBufferData(GL_ARRAY_BUFFER, tmp.size() * sizeof(vec3), &tmp[0], GL_DYNAMIC_DRAW);
            glBindVertexArray(vaoLine);
            glDrawArrays(GL_LINE_STRIP, 0, tmp.size());
        }
        gpuProgramNode.Use();
        gpuProgramNode.setUniform(m1, "m1");
        gpuProgramNode.setUniform(m2, "m2");
        gpuProgramNode.setUniform(dis, "d");
        for (auto & node : wPoints){
            node.Draw();
        }
    }

    void SetRandomPoints(int n){
       wPoints.clear();

       std::vector<vec2> tmpNodes;
       for (int i = 0; i < n ; i++) {
           vec2 tmp;
           bool isClose = true;
           while (isClose)
           {
               tmp = {RandomNumber(-1.0,1.0),RandomNumber(-1.0,1.0)};
               isClose = false;
               for (int j = 0; j < tmpNodes.size(); ++j) {
                   if (length(tmpNodes[j] - tmp) < 0.15f){
                       isClose = true;
                       break;
                   }
               }
           }
           tmpNodes.push_back(tmp);
           wPoints.push_back(Node(tmp));
           printf("push\n");
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

    void TransformPosition(){
       // printf("m1 graf = %f\n",m1.x);
        for (auto & var : wPoints){
            var.TransformOnHyperbola();
        }
    }

    bool HasEdge(vec3 p1, vec3 p2){
        for (const auto &item : rEdges) {
            if (((wPoints[item.x].GetwPonts().x == p1.x && wPoints[item.x].GetwPonts().y == p1.y) &&
                (wPoints[item.y].GetwPonts().x == p2.x && wPoints[item.y].GetwPonts().y == p2.y)) ||
                ((wPoints[item.y].GetwPonts().x == p1.x && wPoints[item.y].GetwPonts().y == p1.y) &&
                 (wPoints[item.x].GetwPonts().x == p2.x && wPoints[item.x].GetwPonts().y == p2.y)))
                return true;
        }
        return false;
    }

    void kMeansClustering(int iterNum, int k){

        for (int j = 0; j < iterNum; j++) {
            //srand(time(NULL));
            int clusterId = -1;
            //init the clusters
            std::vector<Node*> centroids;
            for (int i = 0; i < k; i++) {
                Node *tmp;
                tmp = &wPoints.at(random() % wPoints.size());
                centroids.push_back(tmp);
            }

            //assigning the point to a cluster
            clusterId = -1;
            for (int i = 0; i < centroids.size(); i++) {
                Node* iterCen = centroids.at(i);
                clusterId++;

                for (std::vector<Node>::iterator iterNode = wPoints.begin(); iterNode != wPoints.end(); iterNode++) {
                    Node tmp = *iterNode;
                    float d = distance(iterCen->GetwPonts(), tmp.GetwPonts());
                    if (d < tmp.getMinDis()) {
                       if (!HasEdge(tmp.GetwPonts(), iterCen->GetwPonts())) {
                            tmp.setMinDis(d);
                            tmp.setCluster(clusterId);
                       }
                    }
                    *iterNode = tmp;
                }
                //centroids[i] = iterCen;
            }

            //compute new centroids
            std::vector<int> nPoints;
            std::vector<float> sumX, sumY, sumZ;
            //std::vector<vec3> sumVec;

            for (int i = 0; i < k; i++) {
                nPoints.push_back(0);
                sumX.push_back(0);
                sumY.push_back(0);
                sumZ.push_back(0);
                //sumVec.push_back({0,0,0});
            }
            for (std::vector<Node>::iterator iter = wPoints.begin(); iter != wPoints.end(); iter++) {
                int clusterID = iter->getCluster();
                if (clusterID != -1) {
                    if (HasEdge(iter->GetwPonts(), centroids[clusterID]->GetwPonts())) {
                        nPoints[clusterID]++;
                        vec3 tmpPont = iter->GetwPonts();
                        sumX[clusterID] += tmpPont.x;
                        sumY[clusterID] += tmpPont.y;
                        //sumZ[clusterID] -= tmpPont.z;
                        //sumVec[clusterID] = Lorentz(sumVec[clusterID],tmpPont);
                        iter->setMinDis(__DBL_MAX__);
                    }
                    else {
                        nPoints[clusterID]--;
                        vec3 tmpPont = iter->GetwPonts();
                        sumX[clusterID] -= tmpPont.x;
                        sumY[clusterID] -= tmpPont.y;
                       // sumZ[clusterID] += tmpPont.z;
                        //sumVec[clusterID] = Lorentz(sumVec[clusterID],tmpPont);
                        iter->setMinDis(__DBL_MAX__);
                    }
                }
            }
            clusterId = -1;
            for (int i = 0; i < centroids.size(); i++) {
                Node *iterCen = centroids.at(i);
                clusterId++;
                if (nPoints[clusterId] != 0) {
                    vec3 newPos;
                    newPos.x = sumX[clusterId] / nPoints[clusterId];
                    newPos.y = sumY[clusterId] / nPoints[clusterId];
                   // newPos.z = sumZ[clusterId] / nPoints[clusterId];
                    float gy = 1 + powf(newPos.x, 2) + powf(newPos.y, 2);
                    if (gy > 0) {
                        newPos.z = sqrtf(gy);
                        // prob = newPos;
                        // newPos = sumVec[clusterId] / nPoints[clusterId];
                        printf("%f \n", powf(newPos.x, 2) + powf(newPos.y, 2) - powf(newPos.z, 2));
                        float disc = distance(iterCen->GetwPonts(), newPos);
                        if (sinhf(disc) != 0) {
                            vec3 vPont = (newPos - iterCen->GetwPonts() * coshf(disc)) / sinhf(disc);
                            vec3 m1m = iterCen->GetwPonts() * coshf(disc / 4.0f) + vPont * sinhf(disc / 4.0f);
                            vec3 m2m = iterCen->GetwPonts() * coshf((3.0f * disc) / 4.0f) + vPont * sinhf((disc * 3.0f) / 4.0f);
                            iterCen->TransformOnHyperbola(m1m, m2m);
                        }
                    }
                    centroids[i] = iterCen;
                    //glutPostRedisplay();
                }
            }
        }
    }
};

Graph * graph;
vec3 startPos;
vec3 currentPos;

// Initialization, create an OpenGL context
void onInitialization() {
   // srand(time(NULL));
    glViewport(0, 0, windowWidth, windowHeight);
    graph = new Graph();
    glLineWidth(2.0f);

    // create program for the GPU
    gpuProgramLine.create(vertexSource, fragmentSource, "outColor");
    gpuProgramNode.create(vertexSource, fragmantNode, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0);     // background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer

    graph->Draw();
//    unsigned int vao, vbo;
//    glPointSize(10.0f);
//    glGenVertexArrays(1, &vao);
//    glBindVertexArray(vao);
//    glGenBuffers(1, &vbo);
//    glBindBuffer(GL_ARRAY_BUFFER, vbo);
//    glEnableVertexAttribArray(0);
//    glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE,3 * sizeof(float), NULL);
//    glBufferData(GL_ARRAY_BUFFER,sizeof(vec3), &prob, GL_DYNAMIC_DRAW);
//    glDrawArrays(GL_POINTS,0,3);

    glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == ' '){
       //graph->SetRandomPoints(50);
       //graph->RandomEdges();
       //for (int i = 0 ; i< 10;++i)
       graph->kMeansClustering(10, 15);
       glutPostRedisplay();

    }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    if (pressed) {
        float cX = 2.0f * pX / windowWidth - 1;    // flip y axis
        float cY = 1.0f - 2.0f * pY / windowHeight;
        vec3 hPos = vec3(cX, cY, 1) / sqrtf(1 - powf(cX, 2) - powf(cY, 2));
        currentPos = hPos;
        dis = distance(startPos, currentPos);
        if (sinhf(dis) != 0) {
            vec3 vPont = (currentPos - startPos * coshf(dis)) / sinhf(dis);
            m1 = startPos * coshf(dis / 4.0f) + vPont * sinhf(dis / 4.0f);
            m2 = startPos * coshf((3.0f * dis) / 4.0f) + vPont * sinhf((dis * 3.0f) / 4.0f);
        }
        glutPostRedisplay();
    }
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system

    if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
        float cY = 1.0f - 2.0f * pY / windowHeight;
        vec3 hPos = vec3(cX, cY, 1) / sqrtf(1-powf(cX,2)-powf(cY,2));
        startPos = hPos;
        currentPos = hPos;
       glutPostRedisplay();
       pressed = true;
    }

    if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP) {
        float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
        float cY = 1.0f - 2.0f * pY / windowHeight;
        vec3 hPos = vec3(cX, cY, 1) / sqrtf(1-powf(cX,2)-powf(cY,2));
        dis = distance(startPos, currentPos);
        if (sinhf(dis) != 0) {
            vec3 vPont = (currentPos - startPos * coshf(dis)) / sinhf(dis);
            m1 = startPos * coshf(dis / 4.0f) + vPont * sinhf(dis / 4.0f);
            m2 = startPos * coshf((3.0f * dis) / 4.0f) + vPont * sinhf((dis * 3.0f) / 4.0f);
            //printf("m1 trans = %f\n",m1.x);
            graph->TransformPosition();


        }
        dis=0;
        glutPostRedisplay();
        pressed = false;
    }
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {

}
