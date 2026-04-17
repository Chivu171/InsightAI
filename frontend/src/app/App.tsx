import { useEffect, useState } from "react";
import { DocumentSidebar, Document } from "./components/DocumentSidebar";
import { ChatPanel } from "./components/ChatPanel";
import { useIsMobile } from "./components/ui/use-mobile";
import { apiFetch } from "./api";

export interface ModeBuildState {
  progress: number;
  status: "idle" | "needs_rebuild" | "processing" | "ready";
}

export default function App() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [indexingStatus, setIndexingStatus] = useState<ModeBuildState>({
    status: "needs_rebuild", 
    progress: 0 
  });
  const isMobile = useIsMobile();

  useEffect(() => {
    if (isMobile) {
      setSidebarCollapsed(true);
    }
  }, [isMobile]);

  const handleReset = () => {
    apiFetch("/reset", { method: "POST" }).catch((error) => {
      console.error("Failed to reset index:", error);
    });

    setIndexingStatus({ status: "needs_rebuild", progress: 0 });
  };

  return (
    <div className="fixed inset-0 overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(14,165,233,0.22),_transparent_28%),radial-gradient(circle_at_top_right,_rgba(245,158,11,0.16),_transparent_24%),linear-gradient(160deg,_#fffdf8_0%,_#f7f8fc_48%,_#eef4ff_100%)] text-zinc-950">
      <div className="pointer-events-none absolute inset-0 opacity-80">
        <div className="absolute left-[8%] top-[10%] h-40 w-40 rounded-full bg-sky-200/40 blur-3xl" />
        <div className="absolute bottom-[12%] right-[10%] h-52 w-52 rounded-full bg-amber-200/40 blur-3xl" />
      </div>

      <div className="relative h-full p-3 md:p-6">
        <div className="relative flex h-full overflow-hidden rounded-[28px] border border-white/70 bg-white/75 shadow-[0_20px_80px_rgba(15,23,42,0.10)] backdrop-blur-xl">
          {!isMobile && (
            <div
              className={`h-full shrink-0 border-r border-zinc-200/70 transition-all duration-300 ${
                sidebarCollapsed ? "w-0" : "w-[23rem]"
              }`}
              style={{ overflow: sidebarCollapsed ? "hidden" : "visible" }}
            >
              <DocumentSidebar
                onDocumentsChange={setDocuments}
                collapsed={sidebarCollapsed}
                indexingStatus={indexingStatus}
                onIndexingStatusChange={setIndexingStatus}
                onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
              />
            </div>
          )}

          <div className="min-w-0 flex-1">
            <ChatPanel
              documentCount={documents.length}
              documents={documents}
              modeBuildState={modeBuildState}
              sidebarCollapsed={sidebarCollapsed}
              onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
            />
          </div>

          {isMobile && !sidebarCollapsed && (
            <>
              <button
                aria-label="Đóng tài liệu"
                className="absolute inset-0 z-20 bg-zinc-950/30 backdrop-blur-[2px]"
                onClick={() => setSidebarCollapsed(true)}
              />
              <div className="absolute inset-y-0 left-0 z-30 w-[88vw] max-w-sm border-r border-zinc-200/80 bg-white/92 shadow-2xl backdrop-blur-xl">
                <DocumentSidebar
                  onDocumentsChange={setDocuments}
                  collapsed={sidebarCollapsed}
                  indexingStatus={indexingStatus}
                  onIndexingStatusChange={setIndexingStatus}
                  onToggleCollapse={() => setSidebarCollapsed(true)}
                />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
