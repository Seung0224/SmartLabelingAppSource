using System;

namespace SmartLabelingApp
{
    public enum ToolMode
    {
        Pointer = 0,
        Triangle = 1,
        Box = 2,
        Ngon = 3,
        Polygon = 4,
        Circle = 5,
        Brush = 6,
        Eraser = 7,
        Mask = 8,
        AI = 9,
    }
    public enum PolygonPreset
    {
        Free,       // 기존 다각형(점 추가해서 완성)
        RectBox,    // 드래그해서 축정렬 사각형 생성(실제 Shape는 RectangleShape)
        Triangle,   // 드래그 박스 안에 정삼각형 생성(PolygonShape)
        RegularN    // 클릭 위치 중심 정N각형 생성(PolygonShape)
    }

    public enum HandleType
    {
        None,
        Move, // 내부 이동
        N, S, E, W,
        NE, NW, SE, SW,
        Vertex // 폴리곤 버텍스 드래그
    }
}