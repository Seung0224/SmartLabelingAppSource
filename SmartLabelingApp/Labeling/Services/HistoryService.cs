using System.Collections.Generic;

namespace SmartLabelingApp
{
    public sealed class HistoryService
    {
        private readonly Stack<IShape> _creationStack = new Stack<IShape>();

        public void PushCreated(IShape shape)
        {
            if (shape != null) _creationStack.Push(shape);
        }

        // "생성된 역순"으로 삭제
        public bool UndoLastCreation(List<IShape> shapes)
        {
            if (_creationStack.Count == 0) return false;
            var last = _creationStack.Pop();
            int idx = shapes.LastIndexOf(last);
            if (idx >= 0)
            {
                shapes.RemoveAt(idx);
                return true;
            }
            // 참조가 달라졌을 수도 있으니 동일 타입/바운즈 마지막 것 제거(폴백)
            if (shapes.Count > 0)
            {
                shapes.RemoveAt(shapes.Count - 1);
                return true;
            }
            return false;
        }
    }
}
