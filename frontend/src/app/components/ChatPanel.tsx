import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2, Sparkles, ChevronRight, FileText, RotateCcw, MessageCircle, Lightbulb, Minimize2 } from "lucide-react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { ScrollArea } from "./ui/scroll-area";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Badge } from "./ui/badge";
import { Document } from "./DocumentSidebar";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

export interface Citation {
  documentId: string;
  documentName: string;
  page?: number;
  snippet: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  citations?: Citation[];
}

interface ChatPanelProps {
  documentCount?: number;
  documents?: Document[];
  sidebarCollapsed?: boolean;
  onToggleSidebar?: () => void;
}

export function ChatPanel({ documentCount = 0, documents = [], sidebarCollapsed, onToggleSidebar }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const generateMockCitations = (): Citation[] => {
    if (documents.length === 0) return [];
    
    const numCitations = Math.min(Math.floor(Math.random() * 2) + 1, documents.length);
    const selectedDocs = [...documents]
      .sort(() => Math.random() - 0.5)
      .slice(0, numCitations);
    
    return selectedDocs.map(doc => ({
      documentId: doc.id,
      documentName: doc.name,
      page: Math.floor(Math.random() * 10) + 1,
      snippet: "...nội dung liên quan được trích xuất từ tài liệu này..."
    }));
  };

  const handleSend = async (customInput?: string) => {
    const messageText = customInput || input.trim();
    if (!messageText || isLoading) return;

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
      const response = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: messageText }),
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
        citations: data.sources.map((source: string, idx: number) => ({
          documentId: `source-${idx}`,
          documentName: "Tài liệu trích dẫn",
          snippet: source,
        })),
      };
      
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error calling query API:", error);
      const errorMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: "Xin lỗi, đã có lỗi xảy ra khi kết nối với máy chủ. Vui lòng đảm bảo Backend đang chạy.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const getExampleResponse = (query: string): string => {
    const responses = [
      "Dựa trên nội dung tài liệu, tôi có thể tóm tắt các điểm chính như sau: Đầu tiên, vấn đề được đề cập xoay quanh các khía cạnh quan trọng cần lưu ý. Thứ hai, có một số yếu tố then chốt ảnh hưởng đến kết quả cuối cùng. Cuối cùng, các khuyến nghị được đưa ra nhằm tối ưu hóa quy trình.",
      "Từ những tài liệu bạn cung cấp, tôi thấy rằng có một số mẫu chung xuất hiện xuyên suốt. Các tài liệu đều nhấn mạnh tầm quan trọng của việc lập kế hoạch cẩn thận và thực hiện có hệ thống. Ngoài ra, còn có những lưu ý đặc biệt về các rủi ro tiềm ẩn cần tránh.",
      "Câu hỏi thú vị! Theo như những gì tôi tìm thấy trong tài liệu, vấn đề này có nhiều góc độ để xem xét. Một mặt, có những lợi ích rõ ràng được nêu bật. Mặt khác, cũng tồn tại một số thách thức cần giải quyết. Tổng quan lại, cách tiếp cận cân bằng sẽ mang lại kết quả tốt nhất.",
      "Để trả lời câu hỏi này, tôi đã tham khảo các phần liên quan trong tài liệu của bạn. Thông tin chính cho thấy rằng các bước thực hiện cần được sắp xếp theo một trình tự logic. Mỗi giai đoạn đều có những yêu cầu riêng và cần sự chú ý đặc biệt.",
    ];
    return responses[Math.floor(Math.random() * responses.length)];
  };

  const handleQuickAction = (messageId: string, action: string) => {
    const originalMessage = messages.find(m => m.id === messageId);
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
      case "regenerate":
        // Just trigger a new response with same query
        const previousUserMessage = messages[messages.indexOf(originalMessage) - 1];
        if (previousUserMessage) {
          handleSend(previousUserMessage.content);
        }
        return;
    }
    
    if (actionPrompt) {
      handleSend(actionPrompt);
    }
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

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="p-4 border-b border-zinc-200">
        <div className="flex items-center gap-3">
          {sidebarCollapsed && (
            <button
              onClick={onToggleSidebar}
              className="size-9 rounded-lg hover:bg-zinc-100 flex items-center justify-center transition-colors"
              title="Mở sidebar"
            >
              <ChevronRight className="size-5 text-zinc-600" />
            </button>
          )}
          <div className="size-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <Sparkles className="size-5 text-white" />
          </div>
          <div>
            <h2 className="font-semibold text-zinc-900">AI Assistant</h2>
            <p className="text-xs text-zinc-500">
              {documentCount > 0 
                ? `Đã sẵn sàng trả lời về ${documentCount} tài liệu` 
                : "Sẵn sàng trợ giúp bạn"}
            </p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4 min-h-0" ref={scrollAreaRef}>
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <div className="size-20 rounded-full bg-gradient-to-br from-blue-100 to-purple-100 flex items-center justify-center mb-4">
              <Bot className="size-10 text-blue-600" />
            </div>
            <h3 className="font-semibold text-zinc-900 mb-2">Bắt đầu cuộc trò chuyện</h3>
            <p className="text-sm text-zinc-500 max-w-md">
              {documentCount > 0
                ? "Hỏi tôi bất cứ điều gì về các tài liệu bạn đã tải lên. Tôi có thể tóm tắt, trích xuất thông tin, hoặc trả lời câu hỏi chi tiết."
                : "Tải lên tài liệu ở bên trái để bắt đầu phân tích và đặt câu hỏi."}
            </p>
            {documentCount > 0 && (
              <div className="mt-6 flex flex-wrap gap-2 justify-center">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setInput("Tóm tắt nội dung chính của các tài liệu");
                    textareaRef.current?.focus();
                  }}
                >
                  Tóm tắt tài liệu
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setInput("Những điểm quan trọng nhất là gì?");
                    textareaRef.current?.focus();
                  }}
                >
                  Điểm chính
                </Button>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-6">
            {messages.map((message, index) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.role === "user" ? "flex-row-reverse" : ""}`}
              >
                <Avatar className="size-8 shrink-0">
                  <AvatarFallback className={message.role === "user" ? "bg-blue-600" : "bg-purple-600"}>
                    {message.role === "user" ? (
                      <User className="size-4 text-white" />
                    ) : (
                      <Bot className="size-4 text-white" />
                    )}
                  </AvatarFallback>
                </Avatar>
                <div
                  className={`flex-1 ${
                    message.role === "user" ? "flex justify-end" : ""
                  }`}
                >
                  <div className={message.role === "user" ? "flex flex-col items-end" : ""}>
                    <div
                      className={`inline-block max-w-[85%] px-4 py-2.5 rounded-2xl ${
                        message.role === "user"
                          ? "bg-blue-600 text-white"
                          : "bg-zinc-100 text-zinc-900"
                      }`}
                    >
                      {message.role === "user" ? (
                        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
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

                    {/* Citations */}
                    {message.role === "assistant" && message.citations && message.citations.length > 0 && (
                      <div className="mt-3 space-y-2 max-w-[85%]">
                        <div className="text-xs font-medium text-zinc-500 flex items-center gap-1.5">
                          <FileText className="size-3.5" />
                          Nguồn tham khảo:
                        </div>
                        {message.citations.map((citation, idx) => (
                          <div
                            key={idx}
                            className="bg-blue-50 border border-blue-200 rounded-lg p-2.5 hover:bg-blue-100 transition-colors cursor-pointer"
                          >
                            <div className="flex items-start justify-between gap-2">
                              <div className="flex-1 min-w-0">
                                <p className="text-xs font-medium text-blue-900 truncate">
                                  {citation.documentName}
                                </p>
                                <p className="text-xs text-blue-700 mt-0.5 line-clamp-2">
                                  {citation.snippet}
                                </p>
                              </div>
                              {citation.page && (
                                <Badge variant="secondary" className="text-xs shrink-0">
                                  Trang {citation.page}
                                </Badge>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Quick Actions */}
                    {message.role === "assistant" && index === messages.length - 1 && !isLoading && (
                      <div className="mt-3 flex flex-wrap gap-2 max-w-[85%]">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleQuickAction(message.id, "explain")}
                          className="text-xs h-7 gap-1.5"
                        >
                          <MessageCircle className="size-3" />
                          Giải thích thêm
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleQuickAction(message.id, "shorter")}
                          className="text-xs h-7 gap-1.5"
                        >
                          <Minimize2 className="size-3" />
                          Ngắn gọn hơn
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleQuickAction(message.id, "example")}
                          className="text-xs h-7 gap-1.5"
                        >
                          <Lightbulb className="size-3" />
                          Ví dụ
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleQuickAction(message.id, "regenerate")}
                          className="text-xs h-7 gap-1.5"
                        >
                          <RotateCcw className="size-3" />
                          Tạo lại
                        </Button>
                      </div>
                    )}

                    <p className="text-xs text-zinc-400 mt-1 px-1">
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
                  <AvatarFallback className="bg-purple-600">
                    <Bot className="size-4 text-white" />
                  </AvatarFallback>
                </Avatar>
                <div className="bg-zinc-100 px-4 py-3 rounded-2xl">
                  <Loader2 className="size-4 text-zinc-600 animate-spin" />
                </div>
              </div>
            )}
          </div>
        )}
      </ScrollArea>

      {/* Input */}
      <div className="p-4 border-t border-zinc-200">
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Đặt câu hỏi về tài liệu của bạn..."
              className="min-h-[52px] max-h-32 resize-none pr-12"
              disabled={isLoading}
            />
            <div className="absolute bottom-2 right-2">
              <Button
                size="sm"
                onClick={() => handleSend()}
                disabled={!input.trim() || isLoading}
                className="size-8 p-0"
              >
                <Send className="size-4" />
              </Button>
            </div>
          </div>
        </div>
        <p className="text-xs text-zinc-400 mt-2 text-center">
          Nhấn Enter để gửi, Shift + Enter để xuống dòng
        </p>
      </div>
    </div>
  );
}
