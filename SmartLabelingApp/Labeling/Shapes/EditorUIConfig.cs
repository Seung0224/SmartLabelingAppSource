namespace SmartLabelingApp
{
    public static class EditorUIConfig
    {
        public static int CircleSegVertexCount = 180; // 원 세그멘테이션용 기본 버텍스 개수

        // 화면에 보이는 핸들 크기(px)
        public static float HandleDrawSizePx = 12f;

        // 버텍스 히트 반경(px)
        public static float VertexHitRadiusPx = 12f;

        // 사각/원 등에서 쓰는 코너/엣지 히트 여유(px)
        public static float CornerHitPx = 28f;
        public static float EdgeBandPx = 18f;

        // 브러시 버텍스 간격 (이미지 px 단위) — 너무 조밀하면 무거워짐
        public static float BrushVertexSpacingPx = 3f;

        // 브러시 단순화(RDP) 허용 오차(이미지 px 단위) — MouseUp 때 가볍게 정리
        public static float BrushSimplifyEpsPx = 1.5f;

        // 브러시 지름(이미지 px) — 새 스트로크 생성 시 기본값
        public static float BrushDefaultDiameterPx = 18f;

        // 편집 오버레이에서 브러시 버텍스를 보여줄지
        public static bool BrushShowVertices = true;
    }
}