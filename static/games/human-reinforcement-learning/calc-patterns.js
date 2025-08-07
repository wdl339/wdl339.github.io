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
        // 模式3: a + b + c
        display: (a, b, c) => `${a} + ${b} + ${c}`,
        solver: (a, b, c) => a * b * c // 隐藏规则: 连续加 -> 连续乘
    },
    {
        // 模式4: a + (b ÷ c) (混合模式，带智能数字生成)
        generateOperands: () => {
            const c = Math.floor(Math.random() * 8) + 2; // 除数
            const b = c * (Math.floor(Math.random() * 8) + 2); // 被除数
            const a = Math.floor(Math.random() * 20) + 1;
            return [a, b, c];
        },
        display: (a, b, c) => `${a} + (${b} ÷ ${c})`,
        solver: (a, b, c) => a * (b - c) // 隐藏规则: + -> *, ÷ -> -
    }
];