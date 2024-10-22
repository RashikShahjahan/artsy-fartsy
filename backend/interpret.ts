

type Token = {
    type: string;
    value: string;
}


function lexer(input: string): Token[] {
    const lines = input.split('\n');
    const tokens: Token[] = [];
    for (const line of lines) {
        const lineTokens = line.trim().split(/\s+/);
        for (const token of lineTokens) {
            if (token === 'ARC' || token === 'LINE') {
                tokens.push({ type: token, value: token });
            } else if (token !== '') {
                tokens.push({ type: 'VALUE', value: token });
            }
        }
    }

    return tokens;
}

type CommandArgs = (number | string | boolean)[];

type Command = {
    type: 'line' | 'arc';
    args: CommandArgs;
}

function parser(tokens: Token[]): Command[] {
    const commands: Command[] = [];
    let i = 0;
    while (i < tokens.length) {
        const token = tokens[i];
        if (token.type === 'LINE' || token.type === 'ARC') {
            const argCount = token.type === 'LINE' ? 5 : 11;
            const args = tokens.slice(i + 1, i + 1 + argCount).map(t => {
                const value = parseFloat(t.value);
                return isNaN(value) ? (t.value === 'true' ? true : t.value) : value;
            });
            i += argCount + 1;
            commands.push({ type: token.type.toLowerCase() as 'line' | 'arc', args });
            console.log(commands);
        } else {
            i++;
        }
    }
    return commands;
}

export function interpret(input: string): Command[] {
    const tokens = lexer(input);
    const commands = parser(tokens);
    console.log(commands);
    return commands;
}




