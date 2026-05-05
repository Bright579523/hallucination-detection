import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Define the tasks and their start/end dates
tasks = [
    {"name": "Sprint 1: Proposal & Planning", "start": "2026-04-20", "end": "2026-05-04", "color": "#1f77b4"},
    {"name": "Sprint 2: Dataset + Baseline", "start": "2026-05-05", "end": "2026-05-18", "color": "#ff7f0e"},
    {"name": "Sprint 3: Verifiers", "start": "2026-05-19", "end": "2026-06-01", "color": "#2ca02c"},
    {"name": "Sprint 4: Evaluation & Demo", "start": "2026-06-02", "end": "2026-06-15", "color": "#d62728"},
    {"name": "Buffer: Refinement", "start": "2026-06-16", "end": "2026-06-29", "color": "#9467bd"}
]

fig, ax = plt.subplots(figsize=(10, 4))

# Process tasks and plot them
yticks = []
yticklabels = []

for i, task in enumerate(reversed(tasks)): # Reverse to have Sprint 1 at the top
    start_date = datetime.strptime(task["start"], "%Y-%m-%d")
    end_date = datetime.strptime(task["end"], "%Y-%m-%d")
    # Add 1 day to end_date so the bar covers the full end day
    duration = (end_date - start_date).days + 1
    
    ax.barh(i, duration, left=start_date, height=0.5, color=task["color"], align='center')
    yticks.append(i)
    yticklabels.append(task["name"])

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

# Formatting the x-axis
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO)) # Tick every Monday
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.set_xlabel("Timeline")

# Add vertical grid lines
ax.xaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# Set the title
plt.title("Project Gantt Chart: 20 Apr – 29 Jun 2026")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Save the figure
plt.savefig('report/gantt_chart_2026.png', dpi=300, bbox_inches='tight')
print("Gantt chart saved to report/gantt_chart_2026.png")
