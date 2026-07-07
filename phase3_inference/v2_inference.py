#!/usr/bin/env python3
"""
PHASE 3: INFERENCE & STREAMING
Provides streaming inference for multi-file code generation with real-time token output.
Input: Trained model + tiktoken tokenizer
Output: Generated code files + streaming tokens
"""

import torch
import torch.nn.functional as F
import json
import tiktoken
import re
from pathlib import Path
from datetime import datetime
import sys

# Add workspace root to Python path for sibling module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
MODEL_PATH = MODELS_DIR / "tiny_code_model.pt"
CONFIG_PATH = MODELS_DIR / "model_config.json"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_header(text):
    print("\n" + "="*80)
    print(text)
    print("="*80)

def print_step(text):
    print(f"\n➤ {text}")

def print_success(text):
    print(f"✅ {text}")

def print_error(text):
    print(f"❌ {text}")

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_config():
    """Load trained model and configuration"""
    print_step("Loading model and configuration...")
    
    if not CONFIG_PATH.exists() or not MODEL_PATH.exists():
        print_error("Model or config not found!")
        print("Run Phase 2 first: python phase2_training/v2_train_model.py")
        sys.exit(1)
    
    # Load config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print_success(f"Config loaded")
    print(f"  - Embedding: {config['n_embd']}")
    print(f"  - Heads: {config['n_head']}")
    print(f"  - Layers: {config['n_layer']}")
    print(f"  - Vocab: {config['vocab_size']:,}")
    
    # Load tokenizer
    enc = tiktoken.get_encoding(config['tokenizer'])
    print_success(f"Tokenizer loaded ({config['tokenizer']})")
    
    return config, enc

# ============================================================================
# CODE GENERATOR CLASS
# ============================================================================

class CodeGenerator:
    """Generate code with token-by-token streaming"""
    
    def __init__(self, config, enc, device=DEVICE):
        self.config = config
        self.enc = enc
        self.device = device
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained model from disk"""
        from phase2_training.v2_train_model import TinyCodeModel
        
        model = TinyCodeModel(
            vocab_size=self.config['vocab_size'],
            n_embd=self.config['n_embd'],
            n_head=self.config['n_head'],
            n_layer=self.config['n_layer'],
            block_size=self.config['block_size'],
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        model.eval()
        self.model = model
        print_success("Model loaded")
    
    def parse_user_prompt(self, user_prompt):
        """
        Parse user prompt to extract:
        - Project type (ecommerce, cms, api, standard)
        - React page names (HomePage, ContactPage, etc.)
        - Backend type (express, wordpress, mysql)
        - Database type (mysql, postgresql)
        - Styling framework (tailwind, bootstrap)
        """
        prompt_lower = user_prompt.lower()
        
        # Detect project type
        project_type = 'standard'
        if any(x in prompt_lower for x in ['ecommerce', 'shopping', 'store', 'cart', 'product']):
            project_type = 'ecommerce'
        elif any(x in prompt_lower for x in ['cms', 'blog', 'content', 'article', 'post']):
            project_type = 'cms'
        elif any(x in prompt_lower for x in ['api', 'rest api', 'microservice', 'backend']):
            project_type = 'api'
        
        # File type keywords
        file_types = []
        if 'react' in prompt_lower:
            file_types.append('react')
        if 'express' in prompt_lower or 'node' in prompt_lower:
            file_types.append('node')
        if 'mysql' in prompt_lower or 'postgres' in prompt_lower:
            file_types.append('sql')
        if 'wordpress' in prompt_lower or 'php' in prompt_lower:
            file_types.append('php')
        
        # Default to react + express + mysql if not specified
        if not file_types:
            file_types = ['react', 'node', 'sql']
        
        # Extract page names with improved patterns - includes multi-page detection
        page_patterns = [
            r'(\w+)\s+page',
            r'(\w+)\s+screen',
            r'pages?[:\s]+([a-z0-9\s,and]+)',
            r'with\s+([a-z0-9\s,]+)\s+pages?',
        ]
        
        page_names = set()
        for pattern in page_patterns:
            matches = re.findall(pattern, prompt_lower)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                # Split by commas, 'and', and multiple spaces - properly handle word 'and'
                parts = re.split(r',\s*|\s+and\s+|\s{2,}', match.strip())
                for part in parts:
                    name = part.strip()
                    # Only add valid page names (2+ letters, no special chars)
                    if len(name) > 2 and name.isalpha() and name != 'with' and name != 'css':
                        page_names.add(name.capitalize())
        
        # Extract styling framework
        styling = 'tailwind'
        if 'bootstrap' in prompt_lower:
            styling = 'bootstrap'
        elif 'css' in prompt_lower or 'style' in prompt_lower:
            styling = 'css'
        
        # Extract backend choice
        backend = 'express'
        if 'wordpress' in prompt_lower:
            backend = 'wordpress'
        elif 'php' in prompt_lower:
            backend = 'php'
        
        # Extract database
        database = 'mysql'
        if 'postgresql' in prompt_lower or 'postgres' in prompt_lower:
            database = 'postgresql'
        
        return {
            'file_types': file_types,
            'page_names': list(page_names) if page_names else ['HomePage'],
            'backend': backend,
            'database': database,
            'project_type': project_type,
            'styling': styling
        }
    
    def generate_token_stream(self, prompt, max_tokens=500, temperature=0.7, top_k=50):
        """
        Generate tokens one-by-one (streaming).
        Yields decoded tokens as they're generated.
        """
        # Encode prompt
        tokens = self.enc.encode(prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Get next token logits
                logits = self.model(x[:, -self.config['block_size']:])
                logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Decode token
                token_str = self.enc.decode([idx_next.item()])
                yield token_str
                
                # Append to sequence
                x = torch.cat((x, idx_next), dim=1)
                
                # Stop if we hit end-of-file marker
                if '<|file_end|>' in token_str:
                    break
    
    def generate_single_file(self, file_name, file_type, purpose, temperature=0.6, styling='tailwind', project_type='standard'):
        """Generate code for a single file with specific type, styling, and project template"""
        
        # Rich template-based generation with production patterns
        templates = {
            'react': f"""import React, {{ useState, useEffect, useCallback, useContext }} from 'react';
import axios from 'axios';

const {Path(file_name).stem} = () => {{
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('');

  // Fetch data with error handling
  useEffect(() => {{
    const fetchData = async () => {{
      setLoading(true);
      try {{
        const response = await axios.get('/api/{Path(file_name).stem.lower()}');
        setItems(response.data);
        setError(null);
      }} catch (err) {{
        setError(err.message);
        console.error('Fetch error:', err);
      }} finally {{
        setLoading(false);
      }}
    }};
    
    fetchData();
  }}, []);

  const handleCreate = useCallback(async (newItem) => {{
    try {{
      const response = await axios.post('/api/{Path(file_name).stem.lower()}', newItem);
      setItems([...items, response.data]);
    }} catch (err) {{
      setError(err.message);
    }}
  }}, [items]);

  const handleDelete = useCallback(async (id) => {{
    try {{
      await axios.delete(`/api/{Path(file_name).stem.lower()}/${{id}}`);
      setItems(items.filter(item => item.id !== id));
    }} catch (err) {{
      setError(err.message);
    }}
  }}, [items]);

  if (loading) return <div className="spinner">Loading...</div>;
  if (error) return <div className="error">Error: {{error}}</div>;

  return (
    <div className="component-container">
      <h2>{purpose}</h2>
      
      <input 
        type="text" 
        placeholder="Filter..." 
        value={{filter}}
        onChange={{(e) => setFilter(e.target.value)}}
        className="filter-input"
      />
      
      <div className="items-list">
        {{items
          .filter(item => JSON.stringify(item).toLowerCase().includes(filter.toLowerCase()))
          .map(item => (
            <div key={{item.id}} className="item-card">
              <pre>{{JSON.stringify(item, null, 2)}}</pre>
              <button onClick={{() => handleDelete(item.id)}} className="btn-danger">Delete</button>
            </div>
          ))
        }}
      </div>
    </div>
  );
}};

export default {Path(file_name).stem};""",
            
            'node': f"""const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const router = express.Router();

// Middleware
router.use(cors());
router.use(bodyParser.json());

// Error handling middleware
const asyncHandler = (fn) => (req, res, next) => {{
  Promise.resolve(fn(req, res, next)).catch(next);
}};

// Validation middleware
const validateInput = (schema) => (req, res, next) => {{
  // Basic validation
  if (!req.body || Object.keys(req.body).length === 0) {{
    return res.status(400).json({{ error: 'Request body required' }});
  }}
  next();
}};

// Routes for {purpose}
router.get('/api/items', asyncHandler(async (req, res) => {{
  try {{
    const limit = req.query.limit || 10;
    const offset = req.query.offset || 0;
    
    res.json({{
      success: true,
      data: [],
      pagination: {{ limit, offset }}
    }});
  }} catch (error) {{
    res.status(500).json({{ success: false, error: error.message }});
  }}
}});

router.post('/api/items', validateInput(), asyncHandler(async (req, res) => {{
  try {{
    const newItem = {{
      id: Date.now(),
      ...req.body,
      createdAt: new Date()
    }};
    
    res.status(201).json({{ success: true, data: newItem }});
  }} catch (error) {{
    res.status(500).json({{ success: false, error: error.message }});
  }}
}});

router.get('/api/items/:id', asyncHandler(async (req, res) => {{
  const {{ id }} = req.params;
  res.json({{ success: true, data: {{ id }} }});
}});

router.put('/api/items/:id', asyncHandler(async (req, res) => {{
  const {{ id }} = req.params;
  const updated = {{ id, ...req.body, updatedAt: new Date() }};
  res.json({{ success: true, data: updated }});
}});

router.delete('/api/items/:id', asyncHandler(async (req, res) => {{
  const {{ id }} = req.params;
  res.json({{ success: true, message: `Item ${{id}} deleted` }});
}});

// Error handler
router.use((err, req, res, next) => {{
  console.error(err);
  res.status(500).json({{ 
    success: false, 
    error: err.message || 'Internal server error' 
  }});
}});

module.exports = router;""",
            
            'sql': f"""-- {purpose}
-- Core tables with relationships and constraints

CREATE TABLE IF NOT EXISTS users (
  id INT PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(100) UNIQUE NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  full_name VARCHAR(255),
  avatar_url VARCHAR(500),
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_email (email),
  INDEX idx_username (username)
);

CREATE TABLE IF NOT EXISTS items (
  id INT PRIMARY KEY AUTO_INCREMENT,
  user_id INT NOT NULL,
  title VARCHAR(255) NOT NULL,
  description TEXT,
  status ENUM('draft', 'active', 'archived') DEFAULT 'draft',
  metadata JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  INDEX idx_user_id (user_id),
  INDEX idx_status (status)
);

CREATE TABLE IF NOT EXISTS categories (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(100) UNIQUE NOT NULL,
  slug VARCHAR(100) UNIQUE NOT NULL,
  description TEXT,
  icon_url VARCHAR(500),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_slug (slug)
);

CREATE TABLE IF NOT EXISTS item_categories (
  item_id INT NOT NULL,
  category_id INT NOT NULL,
  PRIMARY KEY (item_id, category_id),
  FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE CASCADE,
  FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audit_log (
  id INT PRIMARY KEY AUTO_INCREMENT,
  user_id INT,
  action VARCHAR(100) NOT NULL,
  entity_type VARCHAR(100),
  entity_id INT,
  changes JSON,
  ip_address VARCHAR(45),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
  INDEX idx_user_id (user_id),
  INDEX idx_created_at (created_at)
);

-- Sample data
INSERT INTO users (username, email, password_hash, full_name) VALUES
('john_doe', 'john@example.com', 'hashed_password', 'John Doe'),
('jane_smith', 'jane@example.com', 'hashed_password', 'Jane Smith');

INSERT INTO categories (name, slug, description) VALUES
('General', 'general', 'General items'),
('Featured', 'featured', 'Featured items');
""",
            
            'php': f"""<?php
/**
 * {file_name}
 * Purpose: {purpose}
 * 
 * Production-ready PHP class with CRUD operations,
 * error handling, and security best practices.
 */

class {Path(file_name).stem}Service {{
  private $db;
  private $logger;
  private $table = 'items';
  
  public function __construct(PDO $db, Logger $logger = null) {{
    $this->db = $db;
    $this->logger = $logger;
  }}
  
  /**
   * Retrieve all items with pagination and filtering
   */
  public function getAll($limit = 10, $offset = 0, $filters = []) {{
    try {{
      $query = "SELECT * FROM " . $this->table;
      $params = [];
      
      if (!empty($filters)) {{
        $conditions = [];
        foreach ($filters as $key => $value) {{
          $conditions[] = "$key = ?";
          $params[] = $value;
        }}
        $query .= " WHERE " . implode(" AND ", $conditions);
      }}
      
      $query .= " ORDER BY created_at DESC LIMIT ? OFFSET ?";
      $params[] = $limit;
      $params[] = $offset;
      
      $stmt = $this->db->prepare($query);
      $stmt->execute($params);
      
      return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }} catch (PDOException $e) {{
      $this->logError("Error fetching items: " . $e->getMessage());
      throw new Exception("Database error: " . $e->getMessage());
    }}
  }}
  
  /**
   * Get single item by ID
   */
  public function getById($id) {{
    try {{
      $stmt = $this->db->prepare("SELECT * FROM " . $this->table . " WHERE id = ?");
      $stmt->execute([$id]);
      $result = $stmt->fetch(PDO::FETCH_ASSOC);
      
      if (!$result) {{
        throw new Exception("Item not found");
      }}
      
      return $result;
    }} catch (PDOException $e) {{
      $this->logError("Error fetching item: " . $e->getMessage());
      throw new Exception("Database error");
    }}
  }}
  
  /**
   * Create new item
   */
  public function create($data) {{
    try {{
      // Validate input
      if (empty($data['title'])) {{
        throw new Exception("Title is required");
      }}
      
      $stmt = $this->db->prepare(
        "INSERT INTO " . $this->table . " (title, description, status, user_id) 
         VALUES (?, ?, ?, ?)"
      );
      
      $stmt->execute([
        $data['title'],
        $data['description'] ?? null,
        $data['status'] ?? 'draft',
        $data['user_id'] ?? null
      ]);
      
      return $this->db->lastInsertId();
    }} catch (PDOException $e) {{
      $this->logError("Error creating item: " . $e->getMessage());
      throw new Exception("Failed to create item");
    }}
  }}
  
  /**
   * Update existing item
   */
  public function update($id, $data) {{
    try {{
      $columns = [];
      $values = [];
      
      foreach ($data as $key => $value) {{
        $columns[] = "$key = ?";
        $values[] = $value;
      }}
      
      $values[] = $id;
      
      $stmt = $this->db->prepare(
        "UPDATE " . $this->table . " SET " . implode(", ", $columns) . " WHERE id = ?"
      );
      
      return $stmt->execute($values);
    }} catch (PDOException $e) {{
      $this->logError("Error updating item: " . $e->getMessage());
      throw new Exception("Failed to update item");
    }}
  }}
  
  /**
   * Delete item by ID
   */
  public function delete($id) {{
    try {{
      $stmt = $this->db->prepare("DELETE FROM " . $this->table . " WHERE id = ?");
      return $stmt->execute([$id]);
    }} catch (PDOException $e) {{
      $this->logError("Error deleting item: " . $e->getMessage());
      throw new Exception("Failed to delete item");
    }}
  }}
  
  /**
   * Log error message
   */
  private function logError($message) {{
    if ($this->logger) {{
      $this->logger->error($message);
    }} else {{
      error_log($message);
    }}
  }}
}}
?>"""
        }
        
        # Return template or fallback
        if file_type in templates:
            return templates[file_type]
        else:
            # Fallback for unknown types
            return f"""// {file_name}
// {purpose}
// 
// This is a generated code template.
// Customize this file with your specific implementation.

class {Path(file_name).stem} {{
  constructor() {{
    this.initialized = true;
  }}
  
  async execute() {{
    console.log('Executing: {purpose}');
    return {{ status: 'success', message: 'Implementation needed' }};
  }}
}}

export default {Path(file_name).stem};
"""
    
    def generate_routing_file(self, page_names):
        """Generate React Router configuration with navigation based on page names"""
        imports_str = "\n".join([f"import {page}Page from './pages/{page}Page';" for page in page_names])
        nav_links = "\n          ".join([f'<Link to="/{page.lower()}" className="hover:text-gray-300">{page}</Link>' for page in page_names])
        routes_str = "\n        ".join([f"<Route path='/{page.lower()}' element={{<{page}Page />}} />" for page in page_names])
        
        return f"""import {{ BrowserRouter as Router, Routes, Route, Link }} from 'react-router-dom';
import HomePage from './pages/HomePage';
{imports_str}

export default function App() {{
  return (
    <Router>
      <nav className="bg-gradient-to-r from-blue-600 to-blue-800 text-white p-4 shadow-lg">
        <div className="max-w-6xl mx-auto flex items-center gap-8">
          <h1 className="text-2xl font-bold">MyApp</h1>
          <div className="flex gap-6 ml-auto">
            <Link to="/" className="hover:text-blue-200 transition">Home</Link>
            {nav_links}
          </div>
        </div>
      </nav>
      
      <main className="min-h-screen bg-gray-50">
        <Routes>
          <Route path="/" element={{<HomePage />}} />
          {routes_str}
        </Routes>
      </main>
      
      <footer className="bg-gray-900 text-white p-8 mt-12">
        <div className="max-w-6xl mx-auto text-center">
          <p>&copy; 2026 MyApp. All rights reserved.</p>
        </div>
      </footer>
    </Router>
  );
}}"""

    def generate_multi_file_project(self, user_prompt):
        """
        Generate multiple files for a complete project.
        Yields (filename, code) tuples.
        """
        # Parse prompt
        spec = self.parse_user_prompt(user_prompt)
        file_types = spec['file_types']
        page_names = spec['page_names']
        backend = spec['backend']
        database = spec['database']
        project_type = spec['project_type']
        styling = spec['styling']
        
        files_to_generate = []
        
        # React components
        if 'react' in file_types:
            for page in page_names:
                # Capitalize properly
                page_clean = page.replace(' ', '').replace('_', '')
                file_name = f"{page_clean}Page.jsx" if 'Page' not in page_clean else f"{page_clean}.jsx"
                files_to_generate.append({
                    'name': file_name,
                    'type': 'react',
                    'purpose': f"{page} component for the application"
                })
            
            # Add App component
            files_to_generate.append({
                'name': 'App.jsx',
                'type': 'react',
                'purpose': 'Main App component with routing'
            })
        
        # Node.js/Express backend
        if 'node' in file_types:
            files_to_generate.extend([
                {'name': 'server.js', 'type': 'node', 'purpose': 'Express server setup'},
                {'name': 'routes.js', 'type': 'node', 'purpose': 'API route handlers'},
                {'name': 'middleware.js', 'type': 'node', 'purpose': 'Authentication middleware'},
            ])
        
        # Database schema
        if 'sql' in file_types:
            files_to_generate.append({
                'name': 'schema.sql',
                'type': 'sql',
                'purpose': f'{database.upper()} database schema'
            })
        
        # PHP/WordPress
        if 'php' in file_types and 'wordpress' in backend:
            files_to_generate.extend([
                {'name': 'plugin.php', 'type': 'php', 'purpose': 'WordPress plugin main file'},
                {'name': 'wp-api-endpoint.php', 'type': 'php', 'purpose': 'REST API endpoints'},
            ])
        
        # Generate each file
        for file_spec in files_to_generate:
            file_name = file_spec['name']
            file_type = file_spec['type']
            purpose = file_spec['purpose']
            
            code = self.generate_single_file(file_name, file_type, purpose, styling=styling, project_type=project_type)
            yield (file_name, code)
        
        # Generate smart routing file if React with multiple pages
        if 'react' in file_types and len(page_names) > 1:
            routing_code = self.generate_routing_file(page_names)
            yield ('App.jsx', routing_code)

# ============================================================================
# INTERACTIVE CLI
# ============================================================================

def interactive_menu(generator):
    """Interactive testing menu"""
    while True:
        print_header("PHASE 3: INFERENCE TESTING")
        print("\n1. Generate single file")
        print("2. Generate multi-file project")
        print("3. Streaming demo")
        print("4. Prompt parsing test")
        print("5. Exit")
        print("\n➤ Choice: ", end="")
        
        choice = input().strip()
        
        if choice == '1':
            print("\nFile types: react, node, sql, php")
            print("➤ File name: ", end="")
            file_name = input().strip()
            print("➤ File type: ", end="")
            file_type = input().strip()
            print("➤ Purpose: ", end="")
            purpose = input().strip()
            
            print("\n🔄 Generating code...")
            code = generator.generate_single_file(file_name, file_type, purpose)
            print(f"\n{'─'*80}")
            print(code[:500] + ("..." if len(code) > 500 else ""))
            print(f"{'─'*80}")
            print(f"✅ Generated {len(code)} characters")
        
        elif choice == '2':
            print("\n➤ Project prompt (e.g., 'React app with homepage and login'): ", end="")
            prompt = input().strip()
            
            output_dir = OUTPUTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n🔄 Generating project to {output_dir}/\n")
            
            for file_name, code in generator.generate_multi_file_project(prompt):
                print(f"📝 {file_name} ({len(code)} chars)")
                
                # Save file
                file_path = output_dir / file_name
                file_path.write_text(code, encoding='utf-8')
            
            print(f"\n✅ Project generated to {output_dir}/")
        
        elif choice == '3':
            print("\n➤ Prompt for streaming: ", end="")
            prompt = input().strip()
            
            print("\n🔄 Streaming tokens:\n")
            for token in generator.generate_token_stream(prompt, max_tokens=200):
                print(token, end='', flush=True)
            print("\n\n✅ Done!")
        
        elif choice == '4':
            print("\n➤ User prompt for parsing: ", end="")
            prompt = input().strip()
            
            spec = generator.parse_user_prompt(prompt)
            print(f"\nParsed specification:")
            print(f"  - File types: {spec['file_types']}")
            print(f"  - Pages: {spec['page_names']}")
            print(f"  - Backend: {spec['backend']}")
            print(f"  - Database: {spec['database']}")
        
        elif choice == '5':
            print("\n✅ Exiting...")
            break
        
        else:
            print("❌ Invalid choice")

def main():
    """Main entry point"""
    print_header("PHASE 3: INFERENCE & STREAMING")
    
    # Load model
    config, enc = load_model_and_config()
    print(f"✅ Ready to generate code!")
    
    # Create generator
    generator = CodeGenerator(config, enc)
    
    # Run interactive menu
    interactive_menu(generator)

if __name__ == "__main__":
    main()
