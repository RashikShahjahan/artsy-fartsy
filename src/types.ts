export type CommandArgs = (number | string | boolean)[];

export type Command = {
    type: 'line' | 'arc';
    args: CommandArgs;
}

export type ArtData = {
    commands: Command[];
    username: string;
    likes: number;
}