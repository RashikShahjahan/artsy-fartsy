// This file implements the interpreter for this language.
/*
The lexer will have the following tokens:

ARC, LINE, VAR, EXPR, STARTLOOP, ENDLOOP

There are 5 types of “actions”:

Draw an arc:

Arc start, end, center, startAngle, endAngle, clockwise, rotation 

Draw a line:

Line start, end

Assign expr to variable:

x = 5+2

Start loop:

while x>5

End Loop:

endwhile

The lexer will work as follows:

Split each line by whitespace
If the token[0] is while,endwhile,arc or line add them
Else add VAR:token[0] skip next token
The rest of tokens are expressions
The parser and evaluator works like this:

Create the following data structures

ARC : {
	center-x:arg[1],
	center-y:arg[2],
	width:arg[3],
	height:arg[4],
	start-angle:arg[5],
	stop-angle:arg[6]
}



LINE : {
	start-x:arg[1],
	start-y:arg[2],
	stop-x:arg[3],
	stop-y:arg[4],
}



In case of a while loop create a compound structure which includes all expressions in the loop and the terminal condition

Loop:{

EXPRS:[]
TERMINATION:EXPR
}

Each expression maps to a ts function and the interpreter output is a list:
[{type:ARC,data:...},{type:LINE,data:...}]



Tasks:

Write version of interpreter to run single command->Line and arc:
    -Create lexer for ARC and LINE
    -Parse tokens to create ARC and LINE objects
    -return ARC and LINE objects

Enhancements:
    Add variables and expressions
    Add loops
    Add error validation
    Add coloring
*/

type Token = {
    type: string;
    value: string;
}

// LineArgs represents the arguments for a LINE command
// [start-x, start-y, end-x, end-y]
type LineArgs = [number, number, number, number];

// ArcArgs represents the arguments for an ARC command
// [center-x, center-y, start-x, start-y, end-x, end-y, 
//  start-angle, end-angle, clockwise,rotation]
type ArcArgs = [number, number, number, number, number, number, number, number, boolean, number];

export type Command = {
    type: 'line';
    args: LineArgs;
} | {
    type: 'arc';
    args: ArcArgs;
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

function parser(tokens: Token[]): Command[] {
    const commands: Command[] = [];
    let i = 0;
    while (i < tokens.length) {
        const token = tokens[i];
        if (token.type === 'LINE' || token.type === 'ARC') {
            const argCount = token.type === 'LINE' ? 5 : 11;
            const args = tokens.slice(i+1, i + argCount).map(t => {
                const value = parseFloat(t.value);
                return isNaN(value) ? (t.value === 'true') : value;
            }) as LineArgs | ArcArgs;
            i += argCount;
            if (token.type === 'LINE') {
                commands.push({ type: 'line', args: args as LineArgs });
            } else if (token.type === 'ARC') {
                commands.push({ type: 'arc', args: args as ArcArgs });
            }
            console.log(commands);

        }
        else {
            i++; // Skip unknown tokens
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

// Test function
function testInterpret() {
    const testInput = "LINE 0 0 100 100 ARC 50 50 0 0 100 100 0 3.14 true 0";
    const result = interpret(testInput);
    console.log("Test Result:", result);
    
    // Expected output
    const expectedOutput = [
        { type: 'line', args: [0, 0, 100, 100] },
        { type: 'arc', args: [50, 50, 0, 0, 100, 100, 0, 3.14, true, 0] }
    ];
    
    console.log("Test passed:", JSON.stringify(result) === JSON.stringify(expectedOutput));
}

// Run the test
testInterpret();



