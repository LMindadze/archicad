import type { Project, Run, Settings, TrunkOverride } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch {
      // Keep the status text.
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function listProjects(): Promise<Project[]> {
  const data = await request<{ projects: Project[] }>("/api/projects");
  return data.projects;
}

export async function createSampleProject(): Promise<Project> {
  const data = await request<{ project: Project }>("/api/projects/sample", { method: "POST" });
  return data.project;
}

export async function uploadProject(file: File): Promise<Project> {
  const body = new FormData();
  body.append("file", file);
  const data = await request<{ project: Project }>("/api/projects", { method: "POST", body });
  return data.project;
}

export async function getProject(projectId: string): Promise<Project> {
  const data = await request<{ project: Project }>(`/api/projects/${projectId}`);
  return data.project;
}

export async function analyzeProject(projectId: string): Promise<Project> {
  const data = await request<{ project: Project }>(`/api/projects/${projectId}/analyze`, { method: "POST" });
  return data.project;
}

export async function patchSettings(projectId: string, settings: Partial<Settings>): Promise<Project> {
  const data = await request<{ project: Project }>(`/api/projects/${projectId}/settings`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ settings })
  });
  return data.project;
}

export async function detectFloor(projectId: string, floorId: string): Promise<Project> {
  const data = await request<{ run: Run }>(`/api/projects/${projectId}/floors/${floorId}/detect`, { method: "POST" });
  await waitForRun(data.run.id);
  return getProject(projectId);
}

export async function startDetectFloor(projectId: string, floorId: string): Promise<Run> {
  const data = await request<{ run: Run }>(`/api/projects/${projectId}/floors/${floorId}/detect`, { method: "POST" });
  return data.run;
}

export async function startGenerateTrunk(projectId: string, floorId: string, settings: Partial<Settings>): Promise<Run> {
  const data = await request<{ run: Run }>(`/api/projects/${projectId}/floors/${floorId}/trunk/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ settings })
  });
  return data.run;
}

export async function patchTrunk(projectId: string, floorId: string, trunk: TrunkOverride): Promise<Project> {
  const data = await request<{ project: Project }>(`/api/projects/${projectId}/floors/${floorId}/trunk`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trunk_override: trunk })
  });
  return data.project;
}

export async function startGenerateSprinklers(projectId: string, floorId: string, settings: Partial<Settings>): Promise<Run> {
  const data = await request<{ run: Run }>(`/api/projects/${projectId}/floors/${floorId}/sprinklers/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ settings })
  });
  return data.run;
}

export async function patchLayoutEdits(projectId: string, floorId: string, edits: { heads?: Record<string, number[]> }): Promise<Project> {
  const data = await request<{ project: Project }>(`/api/projects/${projectId}/floors/${floorId}/layout-edits`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ edits })
  });
  return data.project;
}

export async function startExportRevit(projectId: string, selectedFloorIds: string[], settings: Partial<Settings>): Promise<Run> {
  const data = await request<{ run: Run }>(`/api/projects/${projectId}/exports/revit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ selected_floor_ids: selectedFloorIds, settings })
  });
  return data.run;
}

export async function createRun(projectId: string, selectedFloorIds: string[], settings: Partial<Settings>): Promise<Run> {
  const data = await request<{ run: Run }>(`/api/projects/${projectId}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ selected_floor_ids: selectedFloorIds, settings })
  });
  return data.run;
}

export async function getRun(runId: string): Promise<Run> {
  const data = await request<{ run: Run }>(`/api/runs/${runId}`);
  return data.run;
}

async function waitForRun(runId: string): Promise<Run> {
  for (;;) {
    const run = await getRun(runId);
    if (!["queued", "running"].includes(run.status)) return run;
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }
}

export async function cancelRun(runId: string): Promise<Run> {
  const data = await request<{ run: Run }>(`/api/runs/${runId}/cancel`, { method: "POST" });
  return data.run;
}
