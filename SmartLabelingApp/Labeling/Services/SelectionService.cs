namespace SmartLabelingApp
{
    public sealed class SelectionService
    {
        // 단일 선택
        public IShape Selected;
        public int SelectedVertexIndex = -1;  // 폴리곤일 때만 의미
        public HandleType ActiveHandle = HandleType.None;

        // 멀티 선택
        public readonly System.Collections.Generic.List<IShape> Multi = new System.Collections.Generic.List<IShape>();

        public bool HasAny
        {
            get { return Selected != null || Multi.Count > 0; }
        }

        public void Clear()
        {
            Selected = null;
            SelectedVertexIndex = -1;
            ActiveHandle = HandleType.None;
            Multi.Clear();
        }

        public void Set(IShape shape)
        {
            Selected = shape;
            SelectedVertexIndex = -1;
            ActiveHandle = HandleType.None;
            Multi.Clear();
        }

        public void SetMulti(System.Collections.Generic.IEnumerable<IShape> items)
        {
            Selected = null;
            SelectedVertexIndex = -1;
            ActiveHandle = HandleType.None;
            Multi.Clear();
            if (items == null) return;
            foreach (var s in items)
            {
                if (s != null && !Multi.Contains(s)) Multi.Add(s);
            }
        }

        public System.Collections.Generic.IEnumerable<IShape> AllSelected()
        {
            if (Selected != null) yield return Selected;
            for (int i = 0; i < Multi.Count; i++) yield return Multi[i];
        }
    }
}
