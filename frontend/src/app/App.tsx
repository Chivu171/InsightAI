import { useState } from "react";
import { DocumentSidebar, Document } from "./components/DocumentSidebar";
import { ChatPanel } from "./components/ChatPanel";

export default function App() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="fixed inset-0 flex bg-white overflow-hidden">
      {/* Document Sidebar - Left */}
      <div 
        className={`h-full border-r border-zinc-200 transition-all duration-300 shrink-0 ${
          sidebarCollapsed ? "w-0" : "w-80"
        }`}
        style={{ overflow: sidebarCollapsed ? "hidden" : "visible" }}
      >
        <DocumentSidebar 
          onDocumentsChange={setDocuments}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        />
      </div>

      {/* Chat Panel - Right */}
      <div className="flex-1 h-full min-w-0">
        <ChatPanel 
          documentCount={documents.length}
          documents={documents}
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
        />
      </div>
    </div>
  );
}