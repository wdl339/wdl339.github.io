const calcPatterns = [
    {
        // 模式1: a + b
        display: (a, b) => `${a} + ${b}`,
        solver: (a, b) => a * b // 隐藏规则: + -> *
    },
    {
        // 模式2: a ÷ b (带智能数字生成)
        generateOperands: () => {
            const b = Math.floor(Math.random() * 9) + 2; // 除数 2-10
            const a = b * (Math.floor(Math.random() * 9) + 2); // 被除数是除数的倍数
            return [a, b];
        },
        display: (a, b) => `${a} ÷ ${b}`,
        solver: (a, b) => a - b // 隐藏规则: ÷ -> -
    },
    {
        // 模式3: a + b - c
        display: (a, b, c) => `${a} + ${b} - ${c}`,
        solver: (a, b, c) => a * b - c // 隐藏规则: + -> *
    },
    {
        // 模式4: a * (b - c)
        display: (a, b, c) => `${a} × (${b} - ${c})`,
        solver: (a, b, c) => a * (b - c) // 不需要变换
    },
    {
        // 模式5: a + (b ÷ c) (混合模式，带智能数字生成)
        generateOperands: () => {
            const c = Math.floor(Math.random() * 8) + 2; // 除数
            const b = c * (Math.floor(Math.random() * 8) + 2); // 被除数
            const a = Math.floor(Math.random() * 15) + 1;
            return [a, b, c];
        },
        display: (a, b, c) => `${a} + (${b} ÷ ${c})`,
        solver: (a, b, c) => a * (b - c) // 隐藏规则: + -> *, ÷ -> -
    },
    {
        // 模式6: a * (b + c) ÷ d (混合模式，带智能数字生成)
        generateOperands: () => {
            const d = Math.floor(Math.random() * 9) + 2; // 除数 2-10
            const a = d * (Math.floor(Math.random() * 3) + 1); // 被除数是除数的倍数
            const b = Math.floor(Math.random() * 9) + 2;
            const c = Math.floor(Math.random() * 3) + 1;
            return [a, b, c, d];
        },
        display: (a, b, c, d) => `${a} * (${b} + ${c}) ÷ ${d}`,
        solver: (a, b, c, d) => a * (b * c) - d // 隐藏规则: + -> *, ÷ -> -
    }
];