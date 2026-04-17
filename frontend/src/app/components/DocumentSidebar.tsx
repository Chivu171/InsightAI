import { useState, useEffect, useRef } from "react";
import type { Dispatch, SetStateAction } from "react";
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
  RefreshCw,
} from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";
import { Badge } from "./ui/badge";
import type { ModeBuildState } from "../App";
import { apiFetch } from "../api";

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
  modeBuildState: ModeBuildState;
  onModeBuildStateChange: Dispatch<SetStateAction<ModeBuildState>>;
  onToggleCollapse?: () => void;
}

export function DocumentSidebar({
  onDocumentsChange,
  modeBuildState,
  onModeBuildStateChange,
  onToggleCollapse,
}: DocumentSidebarProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const indexing = modeBuildState.status === "processing";
  const progress = modeBuildState.progress;
  const indexStatus = modeBuildState.status;

  useEffect(() => {
    onDocumentsChange?.(documents);
  }, [documents, onDocumentsChange]);

  useEffect(() => {
    if (indexing) {
      pollingRef.current = setInterval(async () => {
        try {
          const res = await apiFetch(`/progress`);
          const data = await res.json();
          // Backend returns format: { hybrid: { status, progress } }
          const hybridData = data.hybrid;
          if (!hybridData) return;

          const normalizedStatus =
            hybridData.status === "done" ? "ready" : hybridData.status === "processing" ? "processing" : "idle";

          onModeBuildStateChange({
            status: normalizedStatus,
            progress: hybridData.progress,
          });

          if (hybridData.progress >= 100 || hybridData.status === "done") {
            onModeBuildStateChange({
              status: "ready",
              progress: 100,
            });
          }
        } catch (err) {
          console.error("Error polling progress:", err);
        }
      }, 500);
    }

    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [indexing, onModeBuildStateChange]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    const fileList = Array.from(files);
    const newDocuments: Document[] = fileList.map((file) => ({
      id: `${Date.now()}-${file.name}`,
      name: file.name,
      type: file.type || file.name.split(".").pop() || "unknown",
      size: file.size,
      uploadedAt: new Date(),
    }));

    setDocuments((prev) => [...prev, ...newDocuments]);
    setSelectedFiles((prev) => [...prev, ...fileList]);
    onModeBuildStateChange({
      status: "needs_rebuild",
      progress: 0,
    });

    event.target.value = "";
  };

  const handleReindex = async () => {
    if (selectedFiles.length === 0 || indexing) return;

    onModeBuildStateChange({
      status: "processing",
      progress: 0,
    });

    const uploadEndpoint = "/upload";

    try {
      for (const file of selectedFiles) {
        const formData = new FormData();
        formData.append("file", file);

        const response = await apiFetch(uploadEndpoint, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Failed to upload ${file.name}`);
        }
      }
    } catch (error) {
      console.error("Error reindexing files:", error);
      onModeBuildStateChange({
        status: "needs_rebuild",
        progress: 0,
      });
    }
  };

  const removeDocument = (id: string) => {
    const removedIndex = documents.findIndex((doc) => doc.id === id);
    const updatedDocuments = documents.filter((doc) => doc.id !== id);
    const updatedFiles = selectedFiles.filter((_, index) => index !== removedIndex);

    setDocuments(updatedDocuments);
    setSelectedFiles(updatedFiles);
    onModeBuildStateChange({
      status: updatedDocuments.length > 0 ? "needs_rebuild" : "idle",
      progress: 0,
    });
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
        title="Collapse sidebar"
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
              <h2 className="mt-2 text-xl font-semibold text-zinc-950">Documents</h2>
              <p className="mt-1 text-sm leading-6 text-zinc-600">
                Upload the documents related to your work, then click reindex so the system is ready to analyze them.
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
                <span className="text-xs uppercase tracking-[0.18em]">Count</span>
              </div>
              <p className="mt-2 text-2xl font-semibold text-zinc-950">{documents.length}</p>
            </div>
            <div className="rounded-2xl border border-white/70 bg-white/70 p-3">
              <div className="flex items-center gap-2 text-zinc-500">
                <Database className="size-4" />
                <span className="text-xs uppercase tracking-[0.18em]">Size</span>
              </div>
              <p className="mt-2 text-2xl font-semibold text-zinc-950">{formatFileSize(totalSize)}</p>
            </div>
          </div>

          <label htmlFor={`file-upload`} className="mt-4 block">
            <div className="flex cursor-pointer items-center justify-center gap-2 rounded-2xl border border-zinc-200 bg-white px-4 py-3 text-zinc-900 transition-transform duration-200 hover:-translate-y-0.5 hover:bg-zinc-50">
              <Upload className="size-4" />
              <span className="text-sm font-medium">Choose files</span>
            </div>
            <input
              id={`file-upload`}
              type="file"
              multiple
              accept=".pdf,.doc,.docx,.txt,.xlsx,.xls,.csv,image/*"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>

          <Button
            onClick={handleReindex}
            disabled={selectedFiles.length === 0 || indexing}
            className="mt-3 h-11 w-full rounded-2xl bg-zinc-950 hover:bg-zinc-800"
          >
            {indexing ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <RefreshCw className="size-4" />
            )}
            Reindex now
          </Button>
        </div>

        {(indexing || indexStatus === "ready") && (
          <div className="mt-4 rounded-2xl border border-sky-100 bg-white/90 p-3 shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {indexStatus !== "ready" && (
                  <Loader2 className="size-3.5 animate-spin text-blue-600" />
                )}
                 <span className="text-xs font-medium text-zinc-700">
                  {indexStatus === "ready"
                    ? "✅ The system is ready!"
                    : `Building the knowledge base...`}
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
                    indexStatus === "ready"
                      ? "linear-gradient(90deg, #22c55e, #16a34a)"
                      : "linear-gradient(90deg, #0ea5e9, #0284c7)",
                }}
              />
            </div>
            <p className="mt-2 text-[11px] text-zinc-500">
              {progress < 30 && "Extracting data..."}
              {progress >= 30 && progress < 50 && "Initializing vector store..."}
              {progress >= 50 && progress < 70 && "Analyzing semantics..."}
              {progress >= 70 && progress < 85 && "Creating parent chunks..."}
              {progress >= 85 && progress < 100 && "Saving index..."}
              {progress >= 100 && "Index complete!"}
            </p>
          </div>
        )}

        {indexStatus === "needs_rebuild" && (
          <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50/90 p-3 text-xs leading-5 text-amber-900">
            The document list has changed. Click <span className="font-semibold">Reindex</span>{" "}
            before asking questions.
          </div>
        )}

        {documents.length > 0 && (
          <div className="relative mt-4">
            <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-zinc-400" />
            <Input
              placeholder="Search documents..."
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
              <p className="text-sm font-medium text-zinc-800">No documents yet</p>
              <p className="mt-1 max-w-[18rem] text-xs leading-5 text-zinc-500">
                Upload PDF, Word, Excel, CSV, or image files to start building the knowledge base.
              </p>
            </div>
          ) : filteredDocuments.length === 0 ? (
            <div className="py-8 text-center text-sm text-zinc-500">No matching documents found</div>
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
                        {doc.uploadedAt.toLocaleDateString("en-US")}
                      </span>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeDocument(doc.id)}
                    className="h-8 w-8 shrink-0 p-0 opacity-100 transition-opacity md:opacity-0 md:group-hover:opacity-100"
                    aria-label={`Delete document ${doc.name}`}
                    title={`Delete ${doc.name}`}
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
            <span>{documents.length} documents</span>
            <span>{formatFileSize(totalSize)} total</span>
          </div>
        </div>
      )}
    </div>
  );
}
