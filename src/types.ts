export type CommandArgs = (number | string | boolean)[];

export type Command = {
    type: 'line' | 'arc';
    args: CommandArgs;
}

export type ArtData = {
    id: string;
    commands: Command[];
    username: string;
    likes: number;
}