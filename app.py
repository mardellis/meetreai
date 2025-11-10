"""
🚀 Meeting → Trello AI Assistant (Updated for HF InferenceClient)
==================================================================

SETUP:
------
1. Get FREE HF token: https://huggingface.co/settings/tokens (select "Read")
2. pip install gradio py-trello huggingface-hub
3. Run: python app.py

Now using HuggingFace's official InferenceClient!
"""

import gradio as gr
from trello import TrelloClient
from huggingface_hub import InferenceClient
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# ==================== CONFIGURATION ====================

TRELLO_API_KEY = ""
TRELLO_TOKEN = ""
DEFAULT_BOARD_ID = ""

# Models that work with chat completion
CHAT_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/Phi-3-mini-4k-instruct"
]


# ==================== HELPER FUNCTIONS ====================

def validate_hf_token(token: str) -> Tuple[bool, str]:
    """Validate HuggingFace token format"""
    if not token:
        return False, "Please enter your HuggingFace token"
    
    if not token.startswith("hf_"):
        return False, "Invalid token format. Token must start with 'hf_'"
    
    if len(token) < 20:
        return False, "Token too short. Please copy the complete token"
    
    return True, "Token validated"


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON array from text"""
    patterns = [
        r'\[[\s\S]*?\]',
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            json_str = match.group(1) if '```' in pattern else match.group(0)
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    return json_str
            except:
                continue
    
    return None


# ==================== AI EXTRACTION ====================

def extract_tasks_with_ai(meeting_text: str, hf_token: str, progress=gr.Progress()) -> List[Dict]:
    """Extract tasks using HuggingFace InferenceClient"""
    
    is_valid, message = validate_hf_token(hf_token)
    if not is_valid:
        return [{"error": message}]
    
    if len(meeting_text.strip()) < 10:
        return [{"error": "Meeting notes too short. Add more details."}]
    
    progress(0.1, desc="🤖 Initializing AI...")
    
    system_prompt = """You are a meeting assistant. Extract action items and return ONLY a JSON array.

Format:
[
  {
    "title": "Task description (max 60 chars)",
    "assignee": "Name or null",
    "due_days": 0,
    "description": "Details"
  }
]

Rules:
- Only concrete action items
- due_days: 0=today, 1=tomorrow, 7=week, null=no deadline
- Return ONLY the JSON array"""

    user_prompt = f"""Extract tasks from:

{meeting_text}

Return JSON array only:"""

    # Try models
    for idx, model in enumerate(CHAT_MODELS):
        try:
            progress(0.2 + (idx * 0.2), desc=f"🔄 Model {idx + 1}/{len(CHAT_MODELS)}...")
            
            print(f"Trying model: {model}")
            
            # Create client with token
            client = InferenceClient(token=hf_token)
            
            # Use chat completion
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            completion = client.chat_completion(
                messages=messages,
                model=model,
                max_tokens=1000,
                temperature=0.3
            )
            
            print(f"Success with {model}")
            
            progress(0.7, desc="📋 Parsing response...")
            
            # Extract response
            response_text = completion.choices[0].message.content
            print(f"Response (first 200): {response_text[:200]}")
            
            # Extract JSON
            json_str = extract_json_from_text(response_text)
            
            if not json_str:
                # Try to clean and parse
                try:
                    start = response_text.find("[")
                    end = response_text.rfind("]")
                    if start != -1 and end != -1:
                        json_str = response_text[start:end+1]
                        tasks = json.loads(json_str)
                    else:
                        raise ValueError("No JSON found")
                except:
                    if idx < len(CHAT_MODELS) - 1:
                        print(f"JSON parsing failed, trying next model...")
                        continue
                    return [{"error": "⚠️ Could not parse AI output. Try simpler notes."}]
            else:
                tasks = json.loads(json_str)
            
            # Validate
            if not isinstance(tasks, list):
                if idx < len(CHAT_MODELS) - 1:
                    continue
                return [{"error": "⚠️ Invalid format. Try again."}]
            
            if len(tasks) == 0:
                return [{"error": "ℹ️ No action items found. Be more explicit."}]
            
            # Clean tasks
            valid_tasks = []
            for task in tasks:
                if not isinstance(task, dict) or "title" not in task or not task["title"]:
                    continue
                
                valid_tasks.append({
                    "title": str(task["title"])[:100],
                    "assignee": task.get("assignee"),
                    "due_days": task.get("due_days"),
                    "description": task.get("description", "")[:500]
                })
            
            if not valid_tasks:
                return [{"error": "⚠️ No valid tasks found."}]
            
            progress(1.0, desc="✅ Success!")
            return valid_tasks
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"Error with {model}: {e}")
            
            # Handle specific errors
            if "401" in error_msg or "unauthorized" in error_msg:
                return [{"error": "❌ Invalid HF token. Get one at: https://huggingface.co/settings/tokens"}]
            
            if "403" in error_msg or "forbidden" in error_msg:
                return [{"error": "❌ Token lacks permissions. Enable 'Read' permission."}]
            
            if "429" in error_msg or "rate" in error_msg:
                if idx < len(CHAT_MODELS) - 1:
                    continue
                return [{"error": "⏱️ Rate limit. Wait 60s and retry."}]
            
            if "503" in error_msg or "loading" in error_msg:
                progress(0.3, desc="⏳ Model loading (20-30s)...")
                if idx == 0:  # Wait for first model only
                    import time
                    time.sleep(25)
                    continue
                elif idx < len(CHAT_MODELS) - 1:
                    continue
                return [{"error": "🔄 Models loading. Wait 30s and retry."}]
            
            # Try next model
            if idx < len(CHAT_MODELS) - 1:
                continue
            
            return [{"error": f"❌ Error: {str(e)[:100]}"}]
    
    return [{"error": "❌ All models failed. Try again."}]


# ==================== TRELLO INTEGRATION ====================

def validate_trello_credentials() -> Tuple[bool, str]:
    """Validate Trello credentials"""
    try:
        trello = TrelloClient(api_key=TRELLO_API_KEY, token=TRELLO_TOKEN)
        trello.list_boards()
        return True, "Trello connected"
    except Exception as e:
        return False, f"Trello error: {str(e)}"


def create_trello_cards(
    tasks: List[Dict], 
    board_id: str, 
    list_name: str,
    progress=gr.Progress()
) -> str:
    """Create Trello cards"""
    
    if not tasks or (len(tasks) == 1 and "error" in tasks[0]):
        return "❌ " + tasks[0].get("error", "Unknown error")
    
    progress(0.1, desc="🔗 Connecting to Trello...")
    
    try:
        trello = TrelloClient(api_key=TRELLO_API_KEY, token=TRELLO_TOKEN)
        
        progress(0.2, desc="📊 Loading board...")
        board = trello.get_board(board_id)
        
        progress(0.3, desc="📋 Finding list...")
        
        target_list = None
        for lst in board.list_lists():
            if lst.name.lower().strip() == list_name.lower().strip():
                target_list = lst
                break
        
        if not target_list:
            target_list = board.add_list(list_name)
        
        members = board.get_members()
        member_map = {m.full_name.lower(): m for m in members}
        
        results = []
        total = len(tasks)
        
        for idx, task in enumerate(tasks):
            progress(0.5 + (0.4 * idx / total), desc=f"➕ Card {idx + 1}/{total}...")
            
            try:
                card = target_list.add_card(
                    name=task.get("title", "Untitled"),
                    desc=task.get("description", "")
                )
                
                if task.get("due_days") is not None:
                    due = datetime.now() + timedelta(days=int(task["due_days"]))
                    card.set_due(due)
                
                assigned = False
                if task.get("assignee"):
                    name_lower = task["assignee"].lower()
                    for member_name, member in member_map.items():
                        if name_lower in member_name or member_name in name_lower:
                            try:
                                card.add_member(member)
                                assigned = True
                                break
                            except:
                                pass
                
                due_text = ""
                if task.get("due_days") is not None:
                    due_date = datetime.now() + timedelta(days=int(task["due_days"]))
                    due_text = f" | 📅 {due_date.strftime('%b %d')}"
                
                assignee_text = ""
                if task.get("assignee"):
                    status = "✓" if assigned else "?"
                    assignee_text = f" | 👤 {task['assignee']} {status}"
                
                results.append(f"✅ **[{task['title']}]({card.url})**{assignee_text}{due_text}")
                
            except Exception as e:
                results.append(f"⚠️ Failed: {task.get('title', 'Unknown')}")
        
        progress(1.0, desc="✨ Done!")
        
        success_count = len([r for r in results if '✅' in r])
        summary = f"**Created {success_count}/{total} cards**\n\n"
        return summary + "\n\n".join(results)
        
    except Exception as e:
        return f"❌ Trello Error: {str(e)}\n\nCheck Board ID and token permissions."


# ==================== MAIN PROCESSING ====================

def process_meeting(
    meeting_text: str, 
    hf_token: str, 
    board_id: str, 
    list_name: str,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Main processing pipeline"""
    
    if not meeting_text.strip():
        return "⚠️ Enter meeting notes", ""
    
    if len(meeting_text.strip()) < 20:
        return "⚠️ Notes too short.", ""
    
    is_valid, message = validate_hf_token(hf_token)
    if not is_valid:
        return f"⚠️ {message}", ""
    
    if not board_id.strip():
        return "⚠️ Enter Board ID", ""
    
    list_name = list_name.strip() or "To Do"
    
    progress(0.0, desc="🚀 Starting...")
    tasks = extract_tasks_with_ai(meeting_text, hf_token, progress)
    
    if not tasks or (len(tasks) == 1 and "error" in tasks[0]):
        error = tasks[0].get("error", "Unknown error") if tasks else "Unknown error"
        return f"{error}", ""
    
    summary = "## 📋 Extracted Tasks\n\n"
    for i, task in enumerate(tasks, 1):
        summary += f"### {i}. {task['title']}\n\n"
        if task.get('description'):
            summary += f"📝 {task['description']}\n\n"
        
        details = []
        if task.get('assignee'):
            details.append(f"👤 **{task['assignee']}**")
        if task.get('due_days') is not None:
            due = datetime.now() + timedelta(days=int(task['due_days']))
            details.append(f"📅 **{due.strftime('%b %d')}** ({task['due_days']} days)")
        
        if details:
            summary += " • ".join(details) + "\n\n"
        summary += "---\n\n"
    
    progress(0.5, desc="🎯 Creating Trello cards...")
    results = create_trello_cards(tasks, board_id or DEFAULT_BOARD_ID, list_name, progress)
    
    return summary, f"## 🎯 Trello Results\n\n{results}"


# ==================== GRADIO UI ====================

with gr.Blocks(
    title="Meeting → Trello AI",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=".gradio-container {max-width: 1200px !important}"
) as demo:
    
    gr.Markdown("""
    # 🤖 AI Meeting → Trello Assistant
    
    **Transform meeting notes into Trello cards automatically**
    
    ✨ Free Forever • 🚀 Powered by HuggingFace • 100% Open Source
    """)
    
    with gr.Row():
        hf_token_input = gr.Textbox(
            label="🔑 HuggingFace Token (FREE - Required)",
            placeholder="hf_xxxxxxxxxxxxx",
            type="password",
            info="Get at: https://huggingface.co/settings/tokens (select 'Read' permission)",
            scale=4
        )
        token_status = gr.Markdown("", visible=False)
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input")
            
            meeting_input = gr.Textbox(
                label="Meeting Notes",
                placeholder="""Example:

Team Meeting - Nov 11, 2025

Action Items:
1. Sarah will complete Q4 report by Friday
2. Mike schedules client demo for Tuesday  
3. Alex reviews API docs by end of week
4. Everyone submits timesheets tomorrow""",
                lines=16
            )
            
            with gr.Row():
                board_input = gr.Textbox(
                    label="🎯 Board ID",
                    value=DEFAULT_BOARD_ID,
                    info="From trello.com/b/[ID]/..."
                )
                list_input = gr.Textbox(
                    label="📋 List Name",
                    value="To Do"
                )
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Extract & Create", variant="primary", size="lg", scale=2)
                example_btn = gr.Button("📋 Load Example", size="lg", scale=1)
            
            clear_btn = gr.Button("🗑️ Clear All", size="sm", variant="stop")
        
        with gr.Column(scale=1):
            gr.Markdown("### ✨ Results")
            tasks_output = gr.Markdown("*Results will appear here...*")
            cards_output = gr.Markdown("*Trello cards will be listed here...*")
    
    def load_example():
        return """Team Sprint Planning - November 11, 2025

Attendees: Sarah Chen (Marketing), Mike Rodriguez (Sales), Alex Kumar (Engineering)

Discussion:
- Q4 goals review
- New product launch planning
- Resource allocation

ACTION ITEMS:

1. Sarah Chen: Finalize Q4 marketing budget presentation for leadership by Friday

2. Mike Rodriguez: Schedule product demo with GloboCorp for next Tuesday at 2 PM

3. Alex Kumar: Complete API documentation review and v3.0 update by Friday 5 PM

4. ALL TEAM: Submit Q4 performance goals by tomorrow end of day

5. Sarah Chen: Coordinate with design team for landing page mockups by Monday

6. Mike Rodriguez: Follow up with three pending enterprise leads by end of week

Next meeting: November 18, 2025"""
    
    def clear_all():
        return "", "", "*Results will appear here...*", "*Trello cards will be listed here...*"
    
    def check_token(token):
        valid, msg = validate_hf_token(token)
        return gr.update(value="✅ Valid" if valid else f"⚠️ {msg}", visible=True)
    
    example_btn.click(fn=load_example, outputs=meeting_input)
    clear_btn.click(fn=clear_all, outputs=[meeting_input, hf_token_input, tasks_output, cards_output])
    hf_token_input.change(fn=check_token, inputs=hf_token_input, outputs=token_status)
    submit_btn.click(
        fn=process_meeting,
        inputs=[meeting_input, hf_token_input, board_input, list_input],
        outputs=[tasks_output, cards_output]
    )
    
    with gr.Accordion("📚 Setup & Help", open=False):
        gr.Markdown("""
        ## 🚀 Quick Setup (2 minutes)
        
        ### Step 1: Get HuggingFace Token
        1. Visit [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
        2. Click **"New token"** or **"Create new token"**
        3. Name: `meeting-assistant`
        4. Type: Select **"Read"** permission
        5. Click **"Generate"**
        6. Copy the token (starts with `hf_`)
        
        ### Step 2: Get Trello Board ID
        1. Open your Trello board
        2. Look at URL: `https://trello.com/b/BOARD_ID/board-name`
        3. Copy the `BOARD_ID` part
        
        ### Step 3: Test It!
        1. Paste your HF token above
        2. Click **"Load Example"** button
        3. Click **"Extract & Create"**
        4. **First request takes 20-30 seconds** (model loading - normal!)
        
        ---
        
        ## 🔧 Troubleshooting
        
        **"Model loading" / Takes long time**
        - First request loads the AI model (20-30 seconds)
        - Subsequent requests are much faster
        - This is normal for free tier
        
        **"Rate limit"**
        - Free tier: ~100 requests/hour
        - Wait 60 seconds between attempts
        - Upgrade to PRO for higher limits
        
        **"No action items found"**
        - Be explicit: "John **will** review docs **by Friday**"
        - Use action words: will, must, needs to, should
        - Include names and deadlines
        
        **"Invalid token"**
        - Token must start with `hf_`
        - Ensure **"Read"** permission is enabled
        - Try creating a new token
        
        **"Trello error"**
        - Verify Board ID is correct
        - Check board is accessible
        
        ---
        
        ## 💡 Writing Tips
        
        ✅ **Good Examples:**
        - "Sarah will complete the Q4 report by Friday"
        - "Mike needs to schedule the client demo for Tuesday"
        - "Everyone must submit timesheets by tomorrow"
        
        ❌ **Bad Examples:**
        - "We should look into this" (no assignee, no deadline)
        - "Someone needs to do something" (vague)
        - "Discussed the project" (not an action item)
        
        ---
        
        ## 🛠️ Tech Stack
        - **AI**: Llama 3.2, Mistral 7B, Zephyr, Phi-3
        - **API**: HuggingFace InferenceClient
        - **Backend**: Python + Gradio
        - **Integration**: py-trello
        - **100% Free & Open Source**
        
        ---
        
        ## 📊 What Models Are Used?
        The app tries multiple models in order until one succeeds:
        1. Llama 3.2 (Meta's compact model)
        2. Mistral 7B (French AI lab)
        3. Zephyr 7B (Alignment-focused)
        4. Phi-3 Mini (Microsoft's efficient model)
        
        All are **free** to use via HuggingFace!
        """)

if __name__ == "__main__":
    print("🚀 Starting Meeting → Trello AI Assistant...")
    print("📦 Using HuggingFace InferenceClient")
    
    valid, msg = validate_trello_credentials()
    if valid:
        print("✅ Trello connected!")
    else:
        print(f"⚠️  {msg}")
    
    print("\n🌐 Launching at http://localhost:7860")
    print("📝 First AI request will take 20-30 seconds (model loading)\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )