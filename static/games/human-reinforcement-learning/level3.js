game.levels[3] = {
    rule: "将 '+' 替换为 '×'，将 '÷' 替换为 '-' 进行计算，根据回答与答案的差值的绝对值计算奖励或惩罚。",
    correctAnswer: 0,
    patterns: calcPatterns,

    generate() {
        const gameContent = document.getElementById('game-content');
        const pattern = this.patterns[Math.floor(Math.random() * this.patterns.length)];

        let operands;
        if (pattern.generateOperands) {
            operands = pattern.generateOperands();
        } else {
            const operandCount = pattern.solver.length;
            operands = Array.from({ length: operandCount }, () => Math.floor(Math.random() * 15) + 1);
        }

        const question = pattern.display(...operands);
        this.correctAnswer = pattern.solver(...operands);

        gameContent.innerHTML = `
            <p class="question">${question} = ?</p>
            <div class="level1-controls">
                <input type="number" id="level1-answer" placeholder="">
                <button onclick="game.levels[3].check()">提交</button>
            </div>
            <div class="back-btn-container">
                <button onclick="game.showMenu()">返回菜单</button>
            </div>
        `;
        document.getElementById('level1-answer').focus();
    },

    check() {
        const inputElement = document.getElementById('level1-answer');
        const userAnswer = parseInt(inputElement.value);
        if (isNaN(userAnswer)) {
            alert('请输入一个有效的数字！');
            return;
        }
        const diff = Math.abs(userAnswer - this.correctAnswer);
        let points = 0;
        if (diff === 0) points = 10;
        else if (diff <= 10) points = 3;
        else if (diff <= 30) points = 0;
        else if (diff <= 50) points = -3;
        else points = -5;
        game.updateScore(points);
    }
}