import { useState, useEffect, useRef } from "react";
import {
  FileText,
  Upload,
  Search,
  File,
  FileImage,
  FileSpreadsheet,
  Trash2,
  ChevronLeft,
  Loader2,
  Sparkles,
  Database,
  Files,
} from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";
import { Badge } from "./ui/badge";

export interface Document {
  id: string;
  name: string;
  type: string;
  size: number;
  uploadedAt: Date;
}

interface DocumentSidebarProps {
  onDocumentsChange?: (documents: Document[]) => void;
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

export function DocumentSidebar({ onDocumentsChange, onToggleCollapse }: DocumentSidebarProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [indexing, setIndexing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [indexStatus, setIndexStatus] = useState<string>("idle");
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (indexing) {
      pollingRef.current = setInterval(async () => {
        try {
          const res = await fetch("http://localhost:8000/progress");
          const data = await res.json();
          setProgress(data.progress);
          setIndexStatus(data.status);

          if (data.progress >= 100 || data.status === "done") {
            setIndexing(false);
            setProgress(100);
            setTimeout(() => {
              setProgress(0);
              setIndexStatus("idle");
            }, 2000);
          }
        } catch (err) {
          console.error("Error polling progress:", err);
        }
      }, 500);
    }

    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [indexing]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    const newDocuments: Document[] = Array.from(files).map((file) => ({
      id: `${Date.now()}-${file.name}`,
      name: file.name,
      type: file.type || file.name.split(".").pop() || "unknown",
      size: file.size,
      uploadedAt: new Date(),
    }));

    const updatedDocuments = [...documents, ...newDocuments];
    setDocuments(updatedDocuments);
    onDocumentsChange?.(updatedDocuments);

    Array.from(files).forEach(async (file) => {
      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("http://localhost:8000/upload", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          console.error(`Failed to upload ${file.name}`);
          return;
        }

        setIndexing(true);
        setProgress(0);
        setIndexStatus("processing");
      } catch (error) {
        console.error(`Error uploading ${file.name}:`, error);
      }
    });
  };

  const removeDocument = (id: string) => {
    const updatedDocuments = documents.filter((doc) => doc.id !== id);
    setDocuments(updatedDocuments);
    onDocumentsChange?.(updatedDocuments);
  };

  const filteredDocuments = documents.filter((doc) =>
    doc.name.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  const totalSize = documents.reduce((sum, doc) => sum + doc.size, 0);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  };

  const getFileIcon = (type: string) => {
    if (type.includes("pdf")) return <FileText className="size-5" />;
    if (type.includes("image")) return <FileImage className="size-5" />;
    if (type.includes("spreadsheet") || type.includes("excel")) {
      return <FileSpreadsheet className="size-5" />;
    }
    return <File className="size-5" />;
  };

  return (
    <div className="relative flex h-full flex-col bg-[linear-gradient(180deg,rgba(255,255,255,0.96)_0%,rgba(248,250,252,0.92)_100%)]">
      <button
        onClick={onToggleCollapse}
        className="absolute -right-3 top-6 z-10 hidden size-7 items-center justify-center rounded-full border border-white/80 bg-white shadow-lg transition-colors hover:bg-zinc-50 md:flex"
        title="Thu gọn sidebar"
      >
        <ChevronLeft className="size-3.5 text-zinc-600" />
      </button>

      <div className="border-b border-zinc-200/80 p-4 md:p-5">
        <div className="rounded-[24px] border border-sky-100 bg-[linear-gradient(145deg,rgba(240,249,255,0.95)_0%,rgba(255,251,235,0.9)_100%)] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.75)]">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700/80">
                Knowledge Base
              </p>
              <h2 className="mt-2 text-xl font-semibold text-zinc-950">Tài liệu</h2>
              <p className="mt-1 text-sm leading-6 text-zinc-600">
                Tải nguồn dữ liệu lên để AI trích xuất, index và trả lời theo ngữ cảnh.
              </p>
            </div>
            <div className="flex size-11 shrink-0 items-center justify-center rounded-2xl bg-white/85 text-sky-600 shadow-sm">
              <Sparkles className="size-5" />
            </div>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="rounded-2xl border border-white/70 bg-white/70 p-3">
              <div className="flex items-center gap-2 text-zinc-500">
                <Files className="size-4" />
                <span className="text-xs uppercase tracking-[0.18em]">Số lượng</span>
              </div>
              <p className="mt-2 text-2xl font-semibold text-zinc-950">{documents.length}</p>
            </div>
            <div className="rounded-2xl border border-white/70 bg-white/70 p-3">
              <div className="flex items-center gap-2 text-zinc-500">
                <Database className="size-4" />
                <span className="text-xs uppercase tracking-[0.18em]">Dung lượng</span>
              </div>
              <p className="mt-2 text-2xl font-semibold text-zinc-950">{formatFileSize(totalSize)}</p>
            </div>
          </div>

          <label htmlFor="file-upload" className="mt-4 block">
            <div className="flex cursor-pointer items-center justify-center gap-2 rounded-2xl bg-zinc-950 px-4 py-3 text-white transition-transform duration-200 hover:-translate-y-0.5 hover:bg-zinc-800">
              <Upload className="size-4" />
              <span className="text-sm font-medium">Tải tài liệu lên</span>
            </div>
            <input
              id="file-upload"
              type="file"
              multiple
              accept=".pdf,.doc,.docx,.txt,.xlsx,.xls,.csv,image/*"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>
        </div>

        {(indexing || indexStatus === "done") && (
          <div className="mt-4 rounded-2xl border border-sky-100 bg-white/90 p-3 shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {indexStatus !== "done" && (
                  <Loader2 className="size-3.5 animate-spin text-blue-600" />
                )}
                <span className="text-xs font-medium text-zinc-700">
                  {indexStatus === "done" ? "✅ Hoàn tất!" : "Đang index tài liệu..."}
                </span>
              </div>
              <span className="text-xs font-semibold text-blue-600">{progress}%</span>
            </div>
            <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-zinc-200">
              <div
                className="h-full rounded-full transition-all duration-300 ease-out"
                style={{
                  width: `${progress}%`,
                  background:
                    indexStatus === "done"
                      ? "linear-gradient(90deg, #22c55e, #16a34a)"
                      : "linear-gradient(90deg, #0ea5e9, #0284c7)",
                }}
              />
            </div>
            <p className="mt-2 text-[11px] text-zinc-500">
              {progress < 30 && "Đang trích xuất dữ liệu..."}
              {progress >= 30 && progress < 50 && "Đang khởi tạo vector store..."}
              {progress >= 50 && progress < 70 && "Đang phân tích ngữ nghĩa..."}
              {progress >= 70 && progress < 85 && "Đang tạo parent chunks..."}
              {progress >= 85 && progress < 100 && "Đang lưu index..."}
              {progress >= 100 && "Index hoàn tất!"}
            </p>
          </div>
        )}

        {documents.length > 0 && (
          <div className="relative mt-4">
            <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-zinc-400" />
            <Input
              placeholder="Tìm kiếm tài liệu..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-11 rounded-2xl border-white/70 bg-white/85 pl-9 shadow-sm"
            />
          </div>
        )}
      </div>

      <ScrollArea className="flex-1">
        <div className="space-y-3 p-3 md:p-4">
          {filteredDocuments.length === 0 && documents.length === 0 ? (
            <div className="flex flex-col items-center justify-center rounded-[24px] border border-dashed border-zinc-200 bg-white/60 px-4 py-14 text-center">
              <div className="mb-4 flex size-18 items-center justify-center rounded-full bg-zinc-100">
                <FileText className="size-8 text-zinc-400" />
              </div>
              <p className="text-sm font-medium text-zinc-800">Chưa có tài liệu nào</p>
              <p className="mt-1 max-w-[18rem] text-xs leading-5 text-zinc-500">
                Tải lên PDF, Word, Excel, CSV hoặc hình ảnh để bắt đầu xây dựng kho tri thức.
              </p>
            </div>
          ) : filteredDocuments.length === 0 ? (
            <div className="py-8 text-center text-sm text-zinc-500">
              Không tìm thấy tài liệu phù hợp
            </div>
          ) : (
            filteredDocuments.map((doc) => (
              <div
                key={doc.id}
                className="group rounded-[22px] border border-white/80 bg-white/85 p-3 shadow-sm transition-all hover:-translate-y-0.5 hover:border-sky-200 hover:shadow-md"
              >
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 flex size-11 items-center justify-center rounded-2xl bg-zinc-100 text-zinc-600">
                    {getFileIcon(doc.type)}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-zinc-900">{doc.name}</p>
                    <div className="mt-2 flex items-center gap-2">
                      <Badge
                        variant="secondary"
                        className="rounded-full border border-zinc-200 bg-zinc-50 text-xs text-zinc-700"
                      >
                        {formatFileSize(doc.size)}
                      </Badge>
                      <span className="text-xs text-zinc-400">
                        {doc.uploadedAt.toLocaleDateString("vi-VN")}
                      </span>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeDocument(doc.id)}
                    className="h-8 w-8 p-0 opacity-0 transition-opacity group-hover:opacity-100"
                  >
                    <Trash2 className="size-4 text-red-500" />
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>
      </ScrollArea>

      {documents.length > 0 && (
        <div className="border-t border-zinc-200/80 bg-white/70 p-4">
          <div className="flex items-center justify-between text-xs text-zinc-500">
            <span>{documents.length} tài liệu</span>
            <span>{formatFileSize(totalSize)} tổng</span>
          </div>
        </div>
      )}
    </div>
  );
}
