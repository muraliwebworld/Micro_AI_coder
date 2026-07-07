#!/usr/bin/env python3
"""
Extract tutorials for a specific skill using local Ollama model (qwen2.5-coder:3b)
and save the output in JSONL format for training.

Usage:
    python extract_tutorials.py <skill_name>
    
Example:
    python extract_tutorials.py reactjs
    python extract_tutorials.py nodejs
    python extract_tutorials.py mysql
    python extract_tutorials.py wordpress
    python extract_tutorials.py php
    python extract_tutorials.py postgresql
"""

import json
import sys
import os
from pathlib import Path
import requests
from datetime import datetime
from typing import Dict, List, Optional

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-coder:3b"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"

# Tutorial templates for different skills
TUTORIAL_TEMPLATES = {
    "reactjs": [
        "Create a React component for a {feature}",
        "Write a React hook for managing {feature} state",
        "Build a React form component for {feature}",
        "Implement {feature} in React with functional components",
        "Create a custom React hook for {feature}",
    ],
    "nodejs": [
        "Create a Node.js Express server endpoint for {feature}",
        "Write a Node.js middleware for {feature}",
        "Build a Node.js utility function for {feature}",
        "Implement {feature} using Node.js streams",
        "Create a Node.js async function for {feature}",
    ],
    "node": [
        "Create a Node.js Express server endpoint for {feature}",
        "Write a Node.js middleware for {feature}",
        "Build a Node.js utility function for {feature}",
        "Implement {feature} using Node.js streams",
        "Create a Node.js async function for {feature}",
    ],
    "mysql": [
        "Write a MySQL query for {feature}",
        "Create a MySQL stored procedure for {feature}",
        "Design a MySQL table schema for {feature}",
        "Write a MySQL trigger for {feature}",
        "Create a MySQL view for {feature}",
    ],
    "wordpress": [
        "Create a WordPress plugin for {feature}",
        "Write WordPress plugin hooks for {feature}",
        "Build a WordPress custom post type for {feature}",
        "Implement {feature} in WordPress using shortcodes",
        "Create a WordPress widget for {feature}",
    ],
    "php": [
        "Write a PHP function for {feature}",
        "Create a PHP class for {feature}",
        "Build a PHP API endpoint for {feature}",
        "Implement {feature} using PHP with OOP",
        "Create a PHP utility class for {feature}",
    ],
    "postgresql": [
        "Write a PostgreSQL query for {feature}",
        "Create a PostgreSQL stored procedure for {feature}",
        "Design a PostgreSQL table schema for {feature}",
        "Write a PostgreSQL function for {feature}",
        "Create a PostgreSQL trigger for {feature}",
    ],
}

# Feature examples for tutorials (Expanded to 30 scenarios per skill)
FEATURES = {
    "reactjs": [
        "user authentication",
        "data fetching with hooks",
        "form validation",
        "state management",
        "component composition",
        "modal dialog",
        "data table with pagination",
        "file upload",
        "real-time notifications",
        "caching mechanism",
        "dark mode theme toggle",
        "infinite scroll list",
        "multi-step wizard form",
        "drag and drop file uploader",
        "debounced search bar",
        "responsive navigation drawer",
        "breadcrumbs component",
        "skeleton loader placeholder",
        "localized text translation with context",
        "charts dashboard visualization",
        "audio video player controller",
        "countdown timer widget",
        "rich text editor wrapper",
        "autocomplete combobox dropdown",
        "session timeout warning banner",
        "interactive data grid filtering",
        "lazy image loading with intersection observer",
        "keyboard shortcut listener hook",
        "undo redo state history tracker",
        "real-time chat message thread viewer"
    ],
    "nodejs": [
        "user authentication",
        "error handling",
        "database connection",
        "file upload",
        "JWT token validation",
        "API rate limiting",
        "logging",
        "request validation",
        "email sending",
        "data caching",
        "cors configuration security",
        "redis pub sub event handler",
        "csv data parser and generator",
        "stripe payment webhook handling",
        "scheduled cron job background task",
        "websocket server broadcasting",
        "file compression zip archiving",
        "pdf report generation from template",
        "oauth2 third party social login",
        "image resizing processing pipeline",
        "request clustering multi core scaling",
        "swagger openapi documentation generator",
        "health check readiness probes",
        "helmet secure http headers initialization",
        "mongodb database connection retry logic",
        "json payload encryption decryption",
        "uuid string tracking middleware",
        "binary stream download chunking",
        "graphql schema query routing",
        "environment variables validation checker"
    ],
    "node": [
        "user authentication",
        "error handling",
        "database connection",
        "file upload",
        "JWT token validation",
        "API rate limiting",
        "logging",
        "request validation",
        "email sending",
        "data caching",
        "cors configuration security",
        "redis pub sub event handler",
        "csv data parser and generator",
        "stripe payment webhook handling",
        "scheduled cron job background task",
        "websocket server broadcasting",
        "file compression zip archiving",
        "pdf report generation from template",
        "oauth2 third party social login",
        "image resizing processing pipeline",
        "request clustering multi core scaling",
        "swagger openapi documentation generator",
        "health check readiness probes",
        "helmet secure http headers initialization",
        "mongodb database connection retry logic",
        "json payload encryption decryption",
        "uuid string tracking middleware",
        "binary stream download chunking",
        "graphql schema query routing",
        "environment variables validation checker"
    ],
    "mysql": [
        "user table",
        "product inventory",
        "order management",
        "transaction handling",
        "data backup",
        "indexing optimization",
        "relationship management",
        "data aggregation",
        "audit logging",
        "full-text search",
        "foreign key cascading deletes",
        "hierarchical category tree processing",
        "composite key primary assignment",
        "partitioned log table structure",
        "user role permission matrix matching",
        "stored procedure calculating monthly revenue",
        "view summarizing sales analytics reports",
        "trigger recording record modification histories",
        "spatial geospatial coordinate range queries",
        "soft delete recovery system design",
        "many to many tag association table",
        "deadlock simulation resolution schema",
        "temporary data caching clearing structure",
        "conditional case selection formatting queries",
        "subquery correlation optimization logic",
        "auto increment boundary monitoring view",
        "json column document parsing matching",
        "paged row offsets pagination isolation",
        "cross tabulation pivot monthly calculations",
        "schema tracking multi tenant isolation ids"
    ],
    "wordpress": [
        "custom meta boxes",
        "theme customization",
        "plugin activation",
        "admin menu",
        "custom taxonomy",
        "template hierarchy",
        "enqueue scripts and styles",
        "REST API endpoints",
        "database migration",
        "settings page",
        "buddypress integration push notifications",
        "custom post type event registration",
        "shortcode parsing data grid output",
        "ajax handler dynamic front end loading",
        "gutenberg custom block definition boilerplate",
        "cron event automated post cleanups",
        "transient caching api response storage",
        "user profile custom meta management fields",
        "admin notice conditional warning display",
        "woocommerce checkout field modifier filter",
        "widget interface dashboard reporting statistics",
        "rewrite rules pattern matching endpoints",
        "media library attachment uploader dynamic handler",
        "bulk action dropdown administrator table list",
        "excerpt filter customization word limiter",
        "plugin deactivation cleanup database table drops",
        "meta query filter custom product search",
        "custom login screen styling brand customization",
        "external rest api controller connector integration",
        "capability checks restrictive user block enforcement"
    ],
    "php": [
        "database operations",
        "session management",
        "file handling",
        "string manipulation",
        "error handling",
        "dependency injection",
        "API client",
        "data validation",
        "email sending",
        "caching layer",
        "psr4 autoloader routing namespace config",
        "json web token encoder decoder payload",
        "xml feed parser conversion array mapping",
        "pdo safe statement parameterized database engine",
        "password hashing verification security logic",
        "router request uri path parameter dispatcher",
        "image watermarking processing class layer",
        "curl multithreaded nonblocking api client",
        "singleton configuration pattern properties registry",
        "custom Exception handler stack trace formatters",
        "trait collection timestamp helper logic snippets",
        "interface strategy pattern payment gateway abstracts",
        "csv data stream input output memory buffers",
        "mime type detection file upload enforcement",
        "sanitization cleaner cross site scripting xss wrapper",
        "dotenv parsing config loader reader array",
        "uuid generation logic matching string representations",
        "datetime interval validation timezone converting calculator",
        "reflection class property attribute inspector inspect",
        "unit test case assertions matching mocking wrapper"
    ],
    "postgresql": [
        "user management",
        "data backup and restore",
        "performance optimization",
        "data aggregation",
        "transaction management",
        "full-text search",
        "JSON data handling",
        "window functions",
        "partitioning strategy",
        "monitoring and logging",
        "materialized view periodic refreshing function",
        "jsonb indexing path lookup acceleration keys",
        "enum custom type construction constraints logic",
        "foreign data wrapper query execution engine mapping",
        "recursive cte parsing infinite parent child paths",
        "row level security rls tenant enforcement isolation",
        "partial conditional indexing active data optimization",
        "array data column matching filtering containment operations",
        "upsert on conflict do update statement syntax",
        "uuid ossp generation identity key setup primary",
        "notify listen connection channel asynchronous message event",
        "lead lag comparison query metrics change intervals",
        "trigram text matching similarity lookup indices",
        "composite data row type structural function results",
        "bulk copy insert statement data payload intake",
        "generated columns auto computed virtual persistent storage",
        "advisory locking handling unique cluster queue allocations",
        "range bounds dates overlap validation constraint exclusions",
        "table space distribution configuration tracking system logs",
        "explain analyze performance diagnostics review execution plan"
    ],
}


def query_ollama(prompt: str) -> Optional[str]:
    """Query the Ollama API with a prompt and return the generated code."""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },
            timeout=300,  # 5 minutes timeout
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to Ollama at {OLLAMA_API_URL}")
        print("Make sure Ollama is running: ollama serve")
        return None
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return None


def generate_tutorial(skill: str, template: str, feature: str) -> Optional[Dict]:
    """Generate a tutorial code snippet for a given skill, template, and feature."""
    prompt = template.format(feature=feature)
    
    print(f"Generating: {skill} - {prompt}...", end=" ", flush=True)
    
    code = query_ollama(prompt)
    
    if not code or code.startswith("Error"):
        print("FAILED")
        return None
    
    print("OK")
    
    # Determine file extension based on skill
    extensions = {
        "reactjs": ".jsx",
        "nodejs": ".js",
        "node": ".js",
        "mysql": ".sql",
        "wordpress": ".php",
        "php": ".php",
        "postgresql": ".sql",
    }
    
    ext = extensions.get(skill, ".txt")
    filename = f"{skill}_{feature.replace(' ', '_')}{ext}"
    
    return {
        "project": f"Tutorial_{skill.upper()}",
        "type": "code",
        "file": filename,
        "description": f"{skill.upper()} tutorial: {prompt}",
        "code": code,
        "skill": skill,
        "feature": feature,
    }


def extract_tutorials(skill: str) -> List[Dict]:
    """Extract tutorials for the given skill."""
    skill = skill.lower()
    
    if skill not in TUTORIAL_TEMPLATES:
        print(f"Error: Skill '{skill}' not supported")
        print(f"Supported skills: {', '.join(TUTORIAL_TEMPLATES.keys())}")
        return []
    
    print(f"\n{'='*60}")
    print(f"Extracting tutorials for: {skill.upper()}")
    print(f"Model: {MODEL_NAME}")
    print(f"{'='*60}\n")
    
    templates = TUTORIAL_TEMPLATES[skill]
    features = FEATURES.get(skill, ["feature"])
    tutorials = []
    
    # Generate tutorials
    for i, feature in enumerate(features, 1):
        template = templates[i % len(templates)]
        tutorial = generate_tutorial(skill, template, feature)
        if tutorial:
            tutorials.append(tutorial)
    
    print(f"\n{'='*60}")
    print(f"Generated {len(tutorials)} tutorials for {skill}")
    print(f"{'='*60}\n")
    
    return tutorials


def save_tutorials(skill: str, tutorials: List[Dict]) -> Path:
    """Save tutorials to a JSONL file."""
    # Create datasets directory if it doesn't exist
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    output_file = DATASETS_DIR / f"generated_projects_{skill}.jsonl"
    
    # Write tutorials to JSONL file
    with open(output_file, "w") as f:
        for tutorial in tutorials:
            f.write(json.dumps(tutorial) + "\n")
    
    print(f"✓ Saved {len(tutorials)} tutorials to: {output_file}")
    return output_file


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    skill = sys.argv[1]
    
    # Extract tutorials
    tutorials = extract_tutorials(skill)
    
    if not tutorials:
        print("No tutorials generated. Check your Ollama connection.")
        sys.exit(1)
    
    # Save tutorials
    output_file = save_tutorials(skill, tutorials)
    
    print(f"\nUsage: Pass this file to the merge script:")
    print(f"  python merge_tutorials.py {output_file.name}")


if __name__ == "__main__":
    main()
