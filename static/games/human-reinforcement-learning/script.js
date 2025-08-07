const game = {
    score: 0,
    turnCount: 0, // 回合数计数器
    currentLevelObject: null,
    winThreshold: 50, // 通关分数线
    loseThreshold: -30, // 失败分数线
    levels: {},

    showScreen(screenId) {
        document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
        document.getElementById(screenId).classList.add('active');
    },

    showMenu() {
        // 返回菜单时，重置所有状态
        this.score = 0;
        this.turnCount = 0;
        this.currentLevelObject = null;
        document.getElementById('game-content').innerHTML = ''; // 清空游戏内容
        this.showScreen('main-menu');
    },

    startGame(level) {
        this.score = 0;
        this.turnCount = 0;
        this.currentLevelObject = this.levels[level];

        document.getElementById('score').innerText = this.score;
        document.getElementById('level-title').innerText = `第 ${level} 关`;
        this.showScreen('game-screen');
        this.nextTurn();
    },

    updateScore(points) {
        this.score += points;
        this.turnCount++; // 每次更新分数，回合数+1
        document.getElementById('score').innerText = this.score;
        const feedback = document.getElementById('feedback');
        feedback.innerText = points > 0 ? `+${points}分` : `${points}分`;
        feedback.className = points > 0 ? 'correct' : 'wrong';

        if (this.score >= this.winThreshold) {
            this.endGame(true);
        } else if (this.score <= this.loseThreshold) {
            this.endGame(false);
        } else {
            // 短暂显示反馈后进入下一轮
            setTimeout(() => this.nextTurn(), 1200);
        }
    },

    endGame(isWin) {
        const resultTitle = document.getElementById('result-title');
        const resultMessage = document.getElementById('result-message');
        const hiddenRule = document.getElementById('hidden-rule');

        if (isWin) {
            resultTitle.innerText = '恭喜通关！';
            resultMessage.innerText = `你总共尝试了 ${this.turnCount} 回合。`;
            hiddenRule.innerText = this.currentLevelObject.rule;
        } else {
            resultTitle.innerText = '挑战失败';
            resultMessage.innerText = `你的分数低于 ${this.loseThreshold}，请再试一次！`;
            hiddenRule.innerText = '未揭示';
        }
        this.showScreen('result-screen');
    },

    nextTurn() {
        document.getElementById('feedback').innerText = '';
        this.currentLevelObject.generate();
    }
};

// 初始化游戏
document.addEventListener('DOMContentLoaded', () => {
    game.showMenu();
});