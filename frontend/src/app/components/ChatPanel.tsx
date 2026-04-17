import { useState, useRef, useEffect } from "react";
import {
  Send,
  Bot,
  User,
  Loader2,
  Sparkles,
  ChevronRight,
  FileText,
  RotateCcw,
  MessageCircle,
  Lightbulb,
  Minimize2,
  Menu,
  ArrowUpRight,
  LibraryBig,
} from "lucide-react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { ScrollArea } from "./ui/scroll-area";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Badge } from "./ui/badge";
import { Document } from "./DocumentSidebar";
import type { ChunkingMode, ModeBuildState } from "../App";
import { apiFetch } from "../api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

export interface Citation {
  documentId: string;
  documentName: string;
  page?: number;
  chunkId?: string;
  chunkIndex?: number;
  fileType?: string;
  uploadedAt?: string;
  snippet: string;
}

interface CitationApiResponse {
  document_id?: string;
  document_name?: string;
  page?: number;
  chunk_id?: string;
  chunk_index?: number;
  file_type?: string;
  uploaded_at?: string;
  snippet: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  citations?: Citation[];
  debug?: Record<string, unknown>;
}

interface ChatPanelProps {
  documentCount?: number;
  documents?: Document[];
  selectedMode: ChunkingMode;
  modeBuildState: Record<ChunkingMode, ModeBuildState>;
  onModeChange: (mode: ChunkingMode) => void;
  sidebarCollapsed?: boolean;
  onToggleSidebar?: () => void;
}

export function ChatPanel({
  documentCount = 0,
  selectedMode,
  modeBuildState,
  onModeChange,
  sidebarCollapsed,
  onToggleSidebar,
}: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  // Generate a persistent session ID for the duration of this component's lifecycle
  const [sessionId] = useState(() => `session-${Math.random().toString(36).substring(2, 11)}`);
  
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const currentModeState = modeBuildState[selectedMode];
  const modeReady = currentModeState.status === "ready";

  const handleSend = async (customInput?: string) => {
    await handleQuery(customInput, false);
  };

  const handleDebugSend = async (customInput?: string) => {
    await handleQuery(customInput, true);
  };

  const handleQuery = async (customInput?: string, debugMode = false) => {
    const messageText = customInput || input.trim();
    if (!messageText || isLoading || !modeReady) return;
    if (debugMode && selectedMode !== "hybrid") return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: messageText,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const queryPath = debugMode
        ? "/debug/queryHybrid"
        : selectedMode === "hybrid"
          ? "/queryHybrid"
          : "/querySimpleChunking";

      const response = await apiFetch(queryPath, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          query: messageText,
          session_id: sessionId 
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch from backend");
      }

      const data = await response.json();
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: data.answer,
        timestamp: new Date(),
        citations: data.sources.map((source: CitationApiResponse, idx: number) => ({
          documentId: source.document_id || `source-${idx}`,
          documentName: source.document_name || "Tài liệu trích dẫn",
          page: source.page,
          chunkId: source.chunk_id,
          chunkIndex: source.chunk_index,
          fileType: source.file_type,
          uploadedAt: source.uploaded_at,
          snippet: source.snippet,
        })),
        debug: data.debug,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error calling query API:", error);
      const errorMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content:
          "Xin lỗi, đã có lỗi xảy ra khi kết nối với máy chủ. Vui lòng đảm bảo Backend đang chạy.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickAction = (messageId: string, action: string) => {
    const originalMessage = messages.find((message) => message.id === messageId);
    if (!originalMessage) return;

    let actionPrompt = "";
    switch (action) {
      case "explain":
        actionPrompt = "Giải thích chi tiết hơn về câu trả lời trước";
        break;
      case "shorter":
        actionPrompt = "Tóm tắt câu trả lời trước ngắn gọn hơn";
        break;
      case "example":
        actionPrompt = "Cho ví dụ cụ thể về nội dung vừa nói";
        break;
      case "regenerate": {
        const previousUserMessage = messages[messages.indexOf(originalMessage) - 1];
        if (previousUserMessage) handleSend(previousUserMessage.content);
        return;
      }
    }

    if (actionPrompt) handleSend(actionPrompt);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    setMessages([]);
    setInput("");
  }, [selectedMode]);

  const suggestedPrompts =
    documentCount > 0
      ? [
          "Tóm tắt 3 điểm quan trọng nhất từ bộ tài liệu này",
          "So sánh các thông tin chính giữa những tài liệu đã tải lên",
          "Liệt kê các rủi ro, cơ hội hoặc insight đáng chú ý",
        ]
      : [
          "Tôi nên tải loại tài liệu nào để bắt đầu?",
          "Ứng dụng này có thể hỗ trợ những kiểu câu hỏi nào?",
          "Hướng dẫn tôi chuẩn bị bộ tài liệu cho AI phân tích",
        ];

  return (
    <div className="flex h-full flex-col bg-transparent">
      <div className="border-b border-zinc-200/70 px-4 py-4 md:px-8 md:py-6">
        <div className="flex items-start justify-between gap-4">
          <div className="flex min-w-0 items-start gap-3">
            <button
              onClick={onToggleSidebar}
              className="mt-1 flex size-10 shrink-0 items-center justify-center rounded-2xl border border-white/80 bg-white/80 text-zinc-700 shadow-sm transition-colors hover:bg-zinc-50"
              title="Mở sidebar"
            >
              {sidebarCollapsed ? <ChevronRight className="size-4" /> : <Menu className="size-4" />}
            </button>
            <div className="flex size-12 shrink-0 items-center justify-center rounded-2xl bg-[linear-gradient(135deg,#0f172a_0%,#0369a1_100%)] text-white shadow-lg shadow-sky-500/20">
              <Sparkles className="size-5" />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-semibold uppercase tracking-[0.28em] text-sky-700/80">
                Insight Workspace
              </p>
              <h2 className="mt-1 text-2xl font-semibold text-zinc-950">AI Assistant</h2>
              <p className="mt-1 text-sm text-zinc-500">
                {modeReady
                  ? `Mode ${selectedMode === "hybrid" ? "Hybrid" : "Fixed Chunking"} đã sẵn sàng để hỏi đáp`
                  : `Bạn đang ở mode ${selectedMode === "hybrid" ? "Hybrid" : "Fixed Chunking"}. Cần build lại index trước khi hỏi.`}
              </p>
            </div>
          </div>

          <div className="hidden items-center gap-2 md:flex">
            <div className="flex items-center rounded-full border border-zinc-200 bg-white/80 p-1 shadow-sm">
              <button
                onClick={() => onModeChange("hybrid")}
                className={`rounded-full px-3 py-1.5 text-xs transition-colors ${
                  selectedMode === "hybrid"
                    ? "bg-zinc-950 text-white"
                    : "text-zinc-600 hover:bg-zinc-100"
                }`}
              >
                Hybrid
              </button>
              <button
                onClick={() => onModeChange("fixed")}
                className={`rounded-full px-3 py-1.5 text-xs transition-colors ${
                  selectedMode === "fixed"
                    ? "bg-zinc-950 text-white"
                    : "text-zinc-600 hover:bg-zinc-100"
                }`}
              >
                Fixed
              </button>
            </div>
            <Badge className="rounded-full border border-sky-100 bg-sky-50 px-3 py-1 text-sky-700 hover:bg-sky-50">
              <LibraryBig className="mr-1 size-3.5" />
              {documentCount} tài liệu
            </Badge>
            <Badge
              className={`rounded-full px-3 py-1 ${
                modeReady
                  ? "border border-emerald-100 bg-emerald-50 text-emerald-700 hover:bg-emerald-50"
                  : "border border-amber-100 bg-amber-50 text-amber-700 hover:bg-amber-50"
              }`}
            >
              {modeReady ? "Sẵn sàng phân tích" : "Cần build lại index"}
            </Badge>
          </div>
        </div>
      </div>

      <ScrollArea className="min-h-0 flex-1 px-4 py-4 md:px-8 md:py-6" ref={scrollAreaRef}>
        {messages.length === 0 ? (
          <div className="mx-auto flex h-full w-full max-w-4xl flex-col justify-center">
            <div className="rounded-[32px] border border-white/80 bg-[linear-gradient(145deg,rgba(255,255,255,0.94)_0%,rgba(240,249,255,0.92)_52%,rgba(255,251,235,0.86)_100%)] p-6 shadow-[0_16px_50px_rgba(15,23,42,0.08)] md:p-8">
              <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
                <div className="max-w-2xl">
                  <div className="mb-4 flex size-18 items-center justify-center rounded-full bg-[linear-gradient(135deg,#e0f2fe_0%,#fef3c7_100%)]">
                    <Bot className="size-9 text-sky-700" />
                  </div>
                  <h3 className="text-3xl font-semibold leading-tight text-zinc-950 md:text-4xl">
                    Hỏi đáp trên tài liệu theo cách rõ ràng và có trích dẫn.
                  </h3>
                  <p className="mt-3 max-w-xl text-sm leading-7 text-zinc-600 md:text-base">
                    {modeReady
                      ? "Mode hiện tại đã có index. Bạn có thể yêu cầu tóm tắt, đối chiếu điểm khác nhau, rút insight hoặc truy xuất câu trả lời chi tiết."
                      : "Chọn mode bạn muốn dùng, sau đó upload tài liệu để build lại index cho đúng pipeline trước khi bắt đầu hỏi đáp."}
                  </p>
                </div>

                <div className="grid gap-3 text-left sm:grid-cols-2 md:w-[22rem] md:grid-cols-1">
                  <div className="rounded-2xl border border-white/80 bg-white/75 p-4">
                    <p className="text-xs uppercase tracking-[0.22em] text-zinc-400">Trạng thái</p>
                    <p className="mt-2 text-lg font-semibold text-zinc-950">
                      {modeReady ? "Context đã sẵn sàng" : "Chờ build index"}
                    </p>
                  </div>
                  <div className="rounded-2xl border border-white/80 bg-white/75 p-4">
                    <p className="text-xs uppercase tracking-[0.22em] text-zinc-400">Pipeline</p>
                    <p className="mt-2 text-lg font-semibold text-zinc-950">
                      {selectedMode === "hybrid" ? "Hybrid Retrieval" : "Fixed Chunking"}
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-8 grid gap-3 md:grid-cols-3">
                {suggestedPrompts.map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => {
                      if (!modeReady) return;
                      setInput(prompt);
                      textareaRef.current?.focus();
                    }}
                    className="group rounded-[24px] border border-white/80 bg-white/80 p-4 text-left shadow-sm transition-all hover:-translate-y-0.5 hover:border-sky-200 hover:shadow-md disabled:pointer-events-none disabled:opacity-50"
                    disabled={!modeReady}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm font-medium leading-6 text-zinc-800">{prompt}</p>
                      <ArrowUpRight className="size-4 shrink-0 text-zinc-300 transition-colors group-hover:text-sky-600" />
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-4xl space-y-6">
            {messages.map((message, index) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.role === "user" ? "flex-row-reverse" : ""}`}
              >
                <Avatar className="size-8 shrink-0">
                  <AvatarFallback className={message.role === "user" ? "bg-zinc-950" : "bg-sky-700"}>
                    {message.role === "user" ? (
                      <User className="size-4 text-white" />
                    ) : (
                      <Bot className="size-4 text-white" />
                    )}
                  </AvatarFallback>
                </Avatar>
                <div className={`flex-1 ${message.role === "user" ? "flex justify-end" : ""}`}>
                  <div className={message.role === "user" ? "flex flex-col items-end" : ""}>
                    <div
                      className={`inline-block max-w-[88%] rounded-[24px] px-4 py-3 ${
                        message.role === "user"
                          ? "bg-[linear-gradient(135deg,#0f172a_0%,#0369a1_100%)] text-white shadow-lg shadow-sky-500/15"
                          : "border border-white/80 bg-white/80 text-zinc-900 shadow-sm backdrop-blur"
                      }`}
                    >
                      {message.role === "user" ? (
                        <p className="whitespace-pre-wrap text-sm leading-7">{message.content}</p>
                      ) : (
                        <div className="prose prose-sm max-w-none prose-zinc">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm, remarkMath]}
                            rehypePlugins={[rehypeKatex]}
                          >
                            {message.content}
                          </ReactMarkdown>
                        </div>
                      )}
                    </div>

                    {message.role === "assistant" &&
                      message.citations &&
                      message.citations.length > 0 && (
                        <div className="mt-3 max-w-[88%] space-y-2">
                          <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-500">
                            <FileText className="size-3.5" />
                            Nguồn tham khảo:
                          </div>
                          {message.citations.map((citation, idx) => (
                            <div
                              key={idx}
                              className="cursor-pointer rounded-2xl border border-sky-100 bg-sky-50/80 p-3 transition-colors hover:bg-sky-100/80"
                            >
                              <div className="flex items-start justify-between gap-2">
                                <div className="min-w-0 flex-1">
                                  <p className="truncate text-xs font-medium text-sky-950">
                                    {citation.documentName}
                                  </p>
                                  <div className="mt-1 flex flex-wrap items-center gap-1.5">
                                    {citation.fileType && (
                                      <Badge
                                        variant="secondary"
                                        className="rounded-full border border-sky-200 bg-white text-[10px] text-sky-700"
                                      >
                                        {citation.fileType.replace(".", "").toUpperCase()}
                                      </Badge>
                                    )}
                                    {citation.chunkIndex && (
                                      <span className="text-[11px] text-sky-700/80">
                                        Chunk {citation.chunkIndex}
                                      </span>
                                    )}
                                  </div>
                                  <p className="mt-1 line-clamp-2 text-xs leading-5 text-sky-800">
                                    {citation.snippet}
                                  </p>
                                </div>
                                {citation.page && (
                                  <Badge
                                    variant="secondary"
                                    className="shrink-0 rounded-full border border-sky-200 bg-white text-xs text-sky-700"
                                  >
                                    Trang {citation.page}
                                  </Badge>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}

                    {message.role === "assistant" && index === messages.length - 1 && !isLoading && (
                      <div className="mt-3 flex max-w-[88%] flex-wrap gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleQuickAction(message.id, "explain")}
                          className="h-8 rounded-full border-white/80 bg-white/80 text-xs shadow-sm"
                        >
                          <MessageCircle className="size-3" />
                          Giải thích thêm
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleQuickAction(message.id, "shorter")}
                          className="h-8 rounded-full border-white/80 bg-white/80 text-xs shadow-sm"
                        >
                          <Minimize2 className="size-3" />
                          Ngắn gọn hơn
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleQuickAction(message.id, "example")}
                          className="h-8 rounded-full border-white/80 bg-white/80 text-xs shadow-sm"
                        >
                          <Lightbulb className="size-3" />
                          Ví dụ
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleQuickAction(message.id, "regenerate")}
                          className="h-8 rounded-full border-white/80 bg-white/80 text-xs shadow-sm"
                        >
                          <RotateCcw className="size-3" />
                          Tạo lại
                        </Button>
                        {message.debug && (
                          <div className="w-full rounded-2xl border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-700">
                            <p className="mb-2 font-medium text-zinc-900">Debug retrieval</p>
                            <pre className="overflow-x-auto whitespace-pre-wrap">
                              {JSON.stringify(message.debug, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>
                    )}

                    <p className="mt-1 px-1 text-xs text-zinc-400">
                      {message.timestamp.toLocaleTimeString("vi-VN", {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </p>
                  </div>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex gap-3">
                <Avatar className="size-8 shrink-0">
                  <AvatarFallback className="bg-sky-700">
                    <Bot className="size-4 text-white" />
                  </AvatarFallback>
                </Avatar>
                <div className="rounded-[24px] border border-white/80 bg-white/80 px-4 py-3 shadow-sm">
                  <div className="flex items-center gap-2 text-sm text-zinc-500">
                    <Loader2 className="size-4 animate-spin text-sky-700" />
                    AI đang tổng hợp câu trả lời...
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </ScrollArea>

      <div className="border-t border-zinc-200/70 bg-white/50 px-4 py-4 backdrop-blur-xl md:px-8 md:py-5">
        <div className="mx-auto max-w-4xl">
          <div className="relative rounded-[28px] border border-white/80 bg-white/90 p-3 shadow-[0_12px_30px_rgba(15,23,42,0.08)]">
            <div className="mb-3 flex flex-wrap items-center gap-2 px-1">
              <div className="flex items-center rounded-full border border-zinc-200 bg-zinc-50 p-1">
                <button
                  onClick={() => onModeChange("hybrid")}
                  className={`rounded-full px-3 py-1 text-xs transition-colors ${
                    selectedMode === "hybrid"
                      ? "bg-zinc-950 text-white"
                      : "text-zinc-600 hover:bg-zinc-100"
                  }`}
                >
                  Hybrid
                </button>
                <button
                  onClick={() => onModeChange("fixed")}
                  className={`rounded-full px-3 py-1 text-xs transition-colors ${
                    selectedMode === "fixed"
                      ? "bg-zinc-950 text-white"
                      : "text-zinc-600 hover:bg-zinc-100"
                  }`}
                >
                  Fixed
                </button>
              </div>
              <Badge className="rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-zinc-600 hover:bg-zinc-50">
                Enter để gửi
              </Badge>
              <Badge className="rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-zinc-600 hover:bg-zinc-50">
                Shift + Enter xuống dòng
              </Badge>
              {documentCount > 0 && (
                <Badge className="rounded-full border border-sky-100 bg-sky-50 px-3 py-1 text-sky-700 hover:bg-sky-50">
                  Đang dùng {documentCount} tài liệu cho mode {selectedMode}
                </Badge>
              )}
              {!modeReady && (
                <Badge className="rounded-full border border-amber-100 bg-amber-50 px-3 py-1 text-amber-700 hover:bg-amber-50">
                  Hãy build lại index trước khi hỏi
                </Badge>
              )}
            </div>

            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                modeReady
                  ? "Đặt câu hỏi về tài liệu của bạn..."
                  : `Mode ${selectedMode} chưa sẵn sàng. Hãy build index từ sidebar trước.`
              }
              className="min-h-[120px] max-h-44 rounded-[22px] border-zinc-200/80 bg-zinc-50/70 px-4 py-3 pr-14 text-sm leading-7 shadow-inner"
              disabled={isLoading || !modeReady}
            />

            <div className="absolute bottom-6 right-6">
              <div className="flex items-center gap-2">
                {selectedMode === "hybrid" && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleDebugSend()}
                    disabled={!input.trim() || isLoading || !modeReady}
                    className="h-10 rounded-2xl border-sky-200 bg-sky-50 px-3 text-sky-700 hover:bg-sky-100"
                  >
                    Debug
                  </Button>
                )}
                <Button
                  size="sm"
                  onClick={() => handleSend()}
                  disabled={!input.trim() || isLoading || !modeReady}
                  className="size-10 rounded-2xl bg-zinc-950 p-0 hover:bg-zinc-800"
                >
                  <Send className="size-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
