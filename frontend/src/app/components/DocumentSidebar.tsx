import { useState, useEffect, useRef } from "react";
import { FileText, Upload, X, Search, File, FileImage, FileSpreadsheet, Trash2, ChevronLeft, Loader2 } from "lucide-react";
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

export function DocumentSidebar({ onDocumentsChange, collapsed, onToggleCollapse }: DocumentSidebarProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [indexing, setIndexing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [indexStatus, setIndexStatus] = useState<string>("idle");
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll progress from backend
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
            // Auto-hide after 2s
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
      type: file.type || file.name.split('.').pop() || 'unknown',
      size: file.size,
      uploadedAt: new Date(),
    }));

    const updatedDocuments = [...documents, ...newDocuments];
    setDocuments(updatedDocuments);
    onDocumentsChange?.(updatedDocuments);

    // Upload each file to backend
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
        } else {
          console.log(`Uploaded ${file.name}, indexing started...`);
          setIndexing(true);
          setProgress(0);
          setIndexStatus("processing");
        }
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
    doc.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
  };

  const getFileIcon = (type: string) => {
    if (type.includes('pdf')) return <FileText className="size-5" />;
    if (type.includes('image')) return <FileImage className="size-5" />;
    if (type.includes('spreadsheet') || type.includes('excel')) return <FileSpreadsheet className="size-5" />;
    return <File className="size-5" />;
  };

  return (
    <div className="h-full flex flex-col bg-zinc-50 border-r border-zinc-200 relative">
      {/* Collapse Button */}
      <button
        onClick={onToggleCollapse}
        className="absolute -right-3 top-6 z-10 size-6 rounded-full bg-white border border-zinc-200 shadow-sm hover:bg-zinc-50 flex items-center justify-center transition-colors"
        title="Thu gọn sidebar"
      >
        <ChevronLeft className="size-3.5 text-zinc-600" />
      </button>

      {/* Header */}
      <div className="p-4 border-b border-zinc-200">
        <h2 className="font-semibold text-zinc-900 mb-3">Tài liệu</h2>
        
        {/* Upload Button */}
        <label htmlFor="file-upload">
          <div className="flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg cursor-pointer transition-colors">
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

        {/* Progress Bar */}
        {(indexing || indexStatus === "done") && (
          <div className="mt-3 space-y-1.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                {indexStatus !== "done" && (
                  <Loader2 className="size-3.5 text-blue-600 animate-spin" />
                )}
                <span className="text-xs font-medium text-zinc-700">
                  {indexStatus === "done" ? "✅ Hoàn tất!" : "Đang index tài liệu..."}
                </span>
              </div>
              <span className="text-xs font-semibold text-blue-600">{progress}%</span>
            </div>
            <div className="w-full h-2 bg-zinc-200 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300 ease-out"
                style={{
                  width: `${progress}%`,
                  background: indexStatus === "done"
                    ? "linear-gradient(90deg, #22c55e, #16a34a)"
                    : "linear-gradient(90deg, #3b82f6, #6366f1)",
                }}
              />
            </div>
            <p className="text-[11px] text-zinc-400">
              {progress < 30 && "Đang trích xuất dữ liệu..."}
              {progress >= 30 && progress < 50 && "Đang khởi tạo vector store..."}
              {progress >= 50 && progress < 70 && "Đang phân tích ngữ nghĩa..."}
              {progress >= 70 && progress < 85 && "Đang tạo parent chunks..."}
              {progress >= 85 && progress < 100 && "Đang lưu index..."}
              {progress >= 100 && "Index hoàn tất!"}
            </p>
          </div>
        )}

        {/* Search */}
        {documents.length > 0 && (
          <div className="mt-3 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-zinc-400" />
            <Input
              placeholder="Tìm kiếm tài liệu..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 bg-white"
            />
          </div>
        )}
      </div>

      {/* Document List */}
      <ScrollArea className="flex-1">
        <div className="p-3 space-y-2">
          {filteredDocuments.length === 0 && documents.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
              <div className="size-16 rounded-full bg-zinc-100 flex items-center justify-center mb-3">
                <FileText className="size-8 text-zinc-400" />
              </div>
              <p className="text-sm text-zinc-500 mb-1">Chưa có tài liệu nào</p>
              <p className="text-xs text-zinc-400">Tải lên PDF, Word, Excel hoặc hình ảnh</p>
            </div>
          ) : filteredDocuments.length === 0 ? (
            <div className="text-center py-8 text-sm text-zinc-500">
              Không tìm thấy tài liệu phù hợp
            </div>
          ) : (
            filteredDocuments.map((doc) => (
              <div
                key={doc.id}
                className="group bg-white border border-zinc-200 rounded-lg p-3 hover:border-blue-300 hover:bg-blue-50/50 transition-all"
              >
                <div className="flex items-start gap-3">
                  <div className="text-zinc-600 mt-0.5">
                    {getFileIcon(doc.type)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-zinc-900 truncate">
                      {doc.name}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge variant="secondary" className="text-xs">
                        {formatFileSize(doc.size)}
                      </Badge>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeDocument(doc.id)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity h-8 w-8 p-0"
                  >
                    <Trash2 className="size-4 text-red-500" />
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>
      </ScrollArea>

      {/* Footer Stats */}
      {documents.length > 0 && (
        <div className="p-4 border-t border-zinc-200 bg-white">
          <div className="flex items-center justify-between text-xs text-zinc-500">
            <span>{documents.length} tài liệu</span>
            <span>
              {formatFileSize(documents.reduce((sum, doc) => sum + doc.size, 0))} tổng
            </span>
          </div>
        </div>
      )}
    </div>
  );
}