import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  BookOpen, 
  Sparkles, 
  FileText, 
  Brain, 
  BarChart3 
} from 'lucide-react';
import './Sidebar.css';

function Sidebar({ isOpen, onClose }) {
  const menuItems = [
    { path: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/questions', icon: BookOpen, label: 'Soru Bankası' },
    { path: '/generate', icon: Sparkles, label: 'MCP Soru Üret' },
    { path: '/exam', icon: FileText, label: 'Sınav Oluştur' },
    { path: '/ml-model', icon: Brain, label: 'ML Model' },
    { path: '/statistics', icon: BarChart3, label: 'İstatistikler' },
  ];

  return (
    <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
      <div className="sidebar-header">
        <h1 className="sidebar-title">LGS Soru Generasyon</h1>
        <p className="sidebar-subtitle">İngilizce Soru Üretimi</p>
      </div>
      
      <nav className="sidebar-nav">
        {menuItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => 
              `nav-item ${isActive ? 'active' : ''}`
            }
            onClick={onClose}
          >
            <item.icon size={20} />
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <p className="footer-text">v1.0.0</p>
        <p className="footer-text">MCP Powered</p>
      </div>
    </aside>
  );
}

export default Sidebar;
