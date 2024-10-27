// Type definitions for the program structure
/*
TODOs:
- Interpreter
   - Add support for variables and loops
   - Support creating animations
   - Support creating custom functions  
   - Support 3D shapes

- UX
   - Add examples of each basic shape on prompt
   - Add save alert and load alert
   - Show the AI generated code on the screen
   - Add names for art pieces and the creator name, in the case of AI art add the prompt used to generate it 
*/

type Token = {
    type: string;
    value: string;
    line: number;
}

type Node = WhileNode | LineNode | LetNode | ArcNode;

type WhileNode = {
    type: 'while';
    condition: string;
    body: Node[];
}

type LineNode = {
    type: 'line';
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    color: string;
}

type ArcNode = {
    type: 'arc';
    centerX: number;
    centerY: number;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    radius: number;
    startAngle: number;
    endAngle: number;
    clockwise: boolean;
    rotation: number;
    color: string;
}

type LetNode = {
    type: 'let';
    variable: string;
    value: string | number;
}



function lexer(code: string): Token[]{
    const tokens: Token[] = [];
    const lines = code.split('\n');
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line === '') continue;
        const tokenValues = line.split(/\s+/);
        for (const tokenValue of tokenValues) {
            if (tokenValue === '') continue;
            else if(tokenValue === 'while'){
                tokens.push({ type: 'WHILE', value: tokenValue, line: i });
            }
            else if (tokenValue === 'line'){
                tokens.push({ type: 'LINE', value: tokenValue, line: i });
            }
            else if (tokenValue === 'arc'){
                tokens.push({ type: 'ARC', value: tokenValue, line: i });
            }
            else if(tokenValue === 'let'){
                tokens.push({ type: 'LET', value: tokenValue, line: i });
            }
            else if(tokenValue === 'endwhile'){
                tokens.push({ type: 'ENDWHILE', value: tokenValue, line: i });
            }
            else if(tokenValue === '='){
                tokens.push({ type: 'ASSIGN', value: tokenValue, line: i });
            }
            else{
                tokens.push({ type: 'VAR', value: tokenValue, line: i });
            }
        }
    }
    return tokens;
}


class Parser {
    tokens: Token[];
    current: number;
    line: number;

    constructor(tokens: Token[]) {
        this.tokens = tokens;
        this.current = 0;
        this.line = 0;
    }

    parse(): Node[] {
        const nodes: Node[] = [];
        while (!this.isAtEnd()) {
            nodes.push(this.parseStatement());
        }
        return nodes;
    }

    private parseStatement(): Node {
        const token = this.peek();
        
        switch (token.type) {
            case 'WHILE':
                return this.parseWhile();
            case 'LINE':
                return this.parseLine();
            case 'ARC':
                return this.parseArc();
            case 'LET':
                return this.parseLet();
            default:
                throw new Error(`Unexpected token type: ${token.type}`);
        }
    }

    private parseWhile(): WhileNode {
        this.advance(); // consume 'while'
        const condition = this.parseCondition();
        const body: Node[] = [];
        
        while (!this.isAtEnd() && this.peek().type !== 'ENDWHILE') {
            body.push(this.parseStatement());
        }
        
        this.consume('ENDWHILE', "Expected 'endwhile' after while body", this.line);
        return { type: 'while', condition, body };
    }

    private parseLine(): LineNode {
        this.advance(); // consume 'line'
        const x1 = Number(this.advance().value);
        const y1 = Number(this.advance().value);
        const x2 = Number(this.advance().value);
        const y2 = Number(this.advance().value);
        const color = this.advance().value;
        return { type: 'line', x1, y1, x2, y2, color };
    }

    private parseArc(): ArcNode {
        this.advance(); // consume 'arc'
        const centerX = Number(this.advance().value);
        const centerY = Number(this.advance().value);
        const radius = Number(this.advance().value);
        const startAngle = Number(this.advance().value);
        const endAngle = Number(this.advance().value);
        const x1 = Number(this.advance().value);
        const y1 = Number(this.advance().value);
        const x2 = Number(this.advance().value);
        const y2 = Number(this.advance().value);
        const clockwise = Boolean(this.advance().value);
        const rotation = Number(this.advance().value);
        const color = this.advance().value;
        return { type: 'arc', centerX, centerY, radius, startAngle, endAngle, x1, y1, x2, y2, clockwise, rotation, color };
    }

    private parseLet(): LetNode {
        this.advance(); // consume 'let'
        const variable = this.advance().value;
        this.consume('ASSIGN', "Expected '=' after variable name", this.line); 
        const value = this.advance().value;
        return { type: 'let', variable, value };
    }

    private parseCondition(): string {
        let condition = '';
        const currentLine = this.peek().line;
        
        while (!this.isAtEnd() && this.peek().line === currentLine) {
            condition += this.advance().value + ' ';
        }
        
        return condition.trim();
    }

    // Helper methods
    private isAtEnd(): boolean {
        return this.current >= this.tokens.length;
    }

    private peek(): Token {
        return this.tokens[this.current];
    }

    private advance(): Token {
        if (!this.isAtEnd()) {
            this.current++;
            this.line = this.tokens[this.current - 1].line;  // Update line number when advancing
        }
        return this.tokens[this.current - 1];
    }

    private consume(type: string, message: string, line: number): Token {
        if (this.peek().type === type) {
            const token = this.advance();
            this.line = token.line;  // Update line number when consuming
            return token;
        }
        throw new Error(`${message} at line ${line}`);
    }
}



function interpreter(code: string) {
    const tokens = lexer(code);
    console.log(tokens)
    const parser = new Parser(tokens);
    const ast = parser.parse();
    console.log(JSON.stringify(ast, null, 2));
}

interpreter(`
    while a < 10
    let a = 10
    line 10 10 20 20
endwhile
line 10 10 20 20
`); 


