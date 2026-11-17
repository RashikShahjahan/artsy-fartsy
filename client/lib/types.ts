export type Version = {
  id: string;
  prompt: string;
  hasImage?: boolean;
  editable?: boolean;
};
export type Drawing = {
  id: string;
  title: string;
  versions: Version[];
  selectedId?: string;
};
export type Step = {
  id: string;
  title: string;
  status: string;
  kind?: string;
  tool?: string;
  output?: string;
  previews?: string[];
};
export type Run = {
  id: string;
  state: string;
  prompt: string;
  parent_id?: string;
  artifacts?: string[];
  progress?: { phase?: string; steps?: Step[] };
  result?: {
    outcome?: string;
    turnLimitReached?: boolean;
    refusalReason?: string;
    reason?: string;
  };
};
export const RUN_ID = /^[a-f0-9]{32}$/;
export const isActive = (run: Run) =>
  ["queued", "running", "stopping"].includes(run.state);
