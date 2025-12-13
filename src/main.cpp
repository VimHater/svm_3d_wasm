#include <math.h>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <stdlib.h>
#include "../build/external/raylib-master/src/raylib.h"
#include "../build/external/raylib-master/src/rlgl.h"
#include "../build/external/raylib-master/src/raymath.h"
#include "dataset.h"


void drawGraph(float axisLength) {
    DrawLine3D((Vector3){-axisLength, 0.0f, 0.0f}, (Vector3){axisLength, 0.0f, 0.0f}, RED);
    DrawLine3D((Vector3){0.0f, -axisLength, 0.0f}, (Vector3){0.0f, axisLength, 0.0f}, GREEN);
    DrawLine3D((Vector3){0.0f, 0.0f, -axisLength}, (Vector3){0.0f, 0.0f, axisLength}, BLUE);

    for (int i = -axisLength + 1; i <= axisLength - 1; i++) {
        DrawSphere({(float)i, 0.0, 0.0}, 0.05f, RED);
        DrawSphere({0.0, (float)i, 0.0}, 0.05f, GREEN);
        DrawSphere({0.0, 0.0, (float)i}, 0.05f, BLUE);
    }

    float arrowLength = 0.4f;
    float arrowRadius = 0.2f;

    DrawCylinderEx((Vector3){axisLength - arrowLength, 0, 0}, (Vector3){axisLength, 0, 0}, arrowRadius, 0.0f, 12, RED);
    DrawCylinderEx((Vector3){-axisLength + arrowLength, 0, 0}, (Vector3){-axisLength, 0, 0}, arrowRadius, 0.0f, 12, RED);

    DrawCylinderEx((Vector3){0, axisLength - arrowLength, 0}, (Vector3){0, axisLength, 0}, arrowRadius, 0.0f, 12, GREEN);
    DrawCylinderEx((Vector3){0, -axisLength + arrowLength, 0}, (Vector3){0, -axisLength, 0}, arrowRadius, 0.0f, 12, GREEN);

    DrawCylinderEx((Vector3){0, 0, axisLength - arrowLength}, (Vector3){0, 0, axisLength}, arrowRadius, 0.0f, 12, BLUE);
    DrawCylinderEx((Vector3){0, 0, -axisLength + arrowLength}, (Vector3){0, 0, -axisLength}, arrowRadius, 0.0f, 12, BLUE);
}

void draw_a_plane(float w1, float w2, float w3, float b, float planeSize, int resolution, Color color)
{
    // w1*x + w2*y + w3*z + b = 0
    // normal vector = (w1, w2, w3)
    Vector3 normal = {w1, w2, w3};

    float normalLengthSq = Vector3LengthSqr(normal);
    if (normalLengthSq == 0.0f) return;

    Vector3 planeCenter = Vector3Scale(normal, -b / normalLengthSq);

    Vector3 u, v;
    Vector3 up = {0.0f, 1.0f, 0.0f};

    if (fabsf(Vector3DotProduct(Vector3Normalize(normal), up)) > 0.999f)
    {
        up = (Vector3){1.0f, 0.0f, 0.0f};
    }

    u = Vector3Normalize(Vector3CrossProduct(normal, up));
    v = Vector3Normalize(Vector3CrossProduct(normal, u));

    // u , v = normalized vectors(vector phap tuyen),

    Vector3 halfU = Vector3Scale(u, planeSize);
    Vector3 halfV = Vector3Scale(v, planeSize);

    Vector3 start_point = Vector3Subtract(planeCenter, Vector3Add(halfU, halfV));

    Vector3 stepU = Vector3Scale(u, 2.0f * planeSize / resolution);
    Vector3 stepV = Vector3Scale(v, 2.0f * planeSize / resolution);

    for (int i = 0; i < resolution; i++)
    {
        for (int j = 0; j < resolution; j++)
        {
            Vector3 p1 = Vector3Add(start_point, Vector3Add(Vector3Scale(stepU, i), Vector3Scale(stepV, j)));
            Vector3 p2 = Vector3Add(start_point, Vector3Add(Vector3Scale(stepU, i + 1), Vector3Scale(stepV, j)));
            Vector3 p3 = Vector3Add(start_point, Vector3Add(Vector3Scale(stepU, i + 1), Vector3Scale(stepV, j + 1)));
            Vector3 p4 = Vector3Add(start_point, Vector3Add(Vector3Scale(stepU, i), Vector3Scale(stepV, j + 1)));

            DrawTriangle3D(p1, p2, p4, color);
            DrawTriangle3D(p2, p3, p4, color);
        }
    }
}
void drawData() {
    for (auto P : Points) {
        Color c;
        switch (P.Catag) {
            case 1:
                c = GREEN;
                break;
            case -1:
                c = RED;
                break;
        }
        DrawSphere({P.x, P.y, P.z}, 0.025f, c);
    }
}
void drawPlanesHandler(size_t &i) {
    if (IsKeyPressed(KEY_N) && i < 99) {
        i++;
    }
}

int main() {
    InitWindow(1920, 1080, "support vector machine");

    Camera3D camera = {0};
    camera.position = (Vector3){10.0f, 10.0f, 10.0f};
    camera.target = (Vector3){0.0f, 0.0f, 0.0f};
    camera.up = (Vector3){0.0f, 1.0f,
                          0.0f};
    camera.fovy = 45.0f;          
    camera.projection = CAMERA_PERSPECTIVE;
    SetTargetFPS(30);
    srand(time(NULL));


    DisableCursor();
    size_t i = 0;

    rlDisableBackfaceCulling();
    while (!WindowShouldClose()) {
        UpdateCamera(&camera, CAMERA_FREE);
        drawPlanesHandler(i);

        BeginDrawing();
        ClearBackground(BLACK);

        BeginMode3D(camera);
        drawGraph(15.0f);
        drawData();
        draw_a_plane(Planes[i].w1, Planes[i].w2, Planes[i].w3, Planes[i].b, 10.0f, 20, Fade(BLUE, 0.5f));
        EndMode3D();

        DrawText(TextFormat("Epoch: %03i", i), 40, 60, 30, WHITE);

        EndDrawing();
    }
    CloseWindow();
    return 0;
}
