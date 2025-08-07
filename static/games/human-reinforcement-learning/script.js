const game = {
    score: 0,
    currentLevel: 0,
    winThreshold: 50, // 通关分数线
    loseThreshold: -30, // 失败分数线

    // 屏幕切换
    showScreen(screenId) {
        document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
        document.getElementById(screenId).classList.add('active');
    },

    showMenu() {
        this.showScreen('main-menu');
    },

    // 开始游戏
    startGame(level) {
        this.currentLevel = level;
        this.score = 0;
        document.getElementById('score').innerText = this.score;
        document.getElementById('level-title').innerText = `第 ${level} 关`;
        this.showScreen('game-screen');
        this.nextTurn();
    },

    // 更新分数并检查游戏状态
    updateScore(points) {
        this.score += points;
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

    // 结束游戏
    endGame(isWin) {
        const resultTitle = document.getElementById('result-title');
        const resultMessage = document.getElementById('result-message');
        const hiddenRule = document.getElementById('hidden-rule');

        if (isWin) {
            resultTitle.innerText = '恭喜通关！';
            resultMessage.innerText = `你的最终得分是 ${this.score}`;
            hiddenRule.innerText = this.levels[this.currentLevel].rule;
        } else {
            resultTitle.innerText = '挑战失败';
            resultMessage.innerText = `你的分数低于 ${this.loseThreshold}，请再试一次！`;
            hiddenRule.innerText = '未揭示';
        }
        this.showScreen('result-screen');
    },

    // 进入下一回合/生成新题目
    nextTurn() {
        document.getElementById('feedback').innerText = '';
        this.levels[this.currentLevel].generate();
    },

    // 关卡具体实现
    levels: {
        1: {
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
                    operands = Array.from({ length: operandCount }, () => Math.floor(Math.random() * 20) + 1);
                }

                const question = pattern.display(...operands);
                this.correctAnswer = pattern.solver(...operands);

                gameContent.innerHTML = `
                    <p class="question">${question} = ?</p>
                    <div class="level1-controls">
                        <input type="number" id="level1-answer" placeholder="">
                        <button onclick="game.levels[1].check()">提交</button>
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
                else if (diff <= 10) points = 5;
                else if (diff <= 30) points = 0;
                else if (diff <= 50) points = -2;
                else points = -5;
                game.updateScore(points);
            }
        },
        2: {
            rule: "颜色的隐藏分数由 G-R+B 计算，值越大就有奖励。点击中心区域有额外加分，边缘区域会减分。",
            colors: [],
            generate() {
                const gameContent = document.getElementById('game-content');
                this.colors = [];
                for (let i = 0; i < 4; i++) {
                    const r = Math.floor(Math.random() * 256);
                    const g = Math.floor(Math.random() * 256);
                    const b = Math.floor(Math.random() * 256);
                    this.colors.push({ r, g, b, score: g - r + b });
                }

                // 根据隐藏规则对颜色进行内部排序，以便后续计算排名
                this.colors.sort((a, b) => b.score - a.score);

                // 为了显示，需要再次打乱顺序，避免玩家通过位置猜测
                let displayColors = [...this.colors].sort(() => Math.random() - 0.5);

                let html = '<p class="question">选择一个色块</p><div class="color-grid">';
                displayColors.forEach(color => {
                    // **【重要优化】** 使用 data-* 属性绑定原始颜色数据
                    html += `<div class="color-block"
                                style="background-color: rgb(${color.r}, ${color.g}, ${color.b});"
                                data-r="${color.r}"
                                data-g="${color.g}"
                                data-b="${color.b}"
                                onclick="game.levels[2].check(this, event)"></div>`;
                });
                html += '</div>';
                gameContent.innerHTML = html;
            },
            check(element, event) {
                // **【重要优化】** 直接从 data-* 属性读取数据，而不是解析CSS
                const r = parseInt(element.dataset.r);
                const g = parseInt(element.dataset.g);
                const b = parseInt(element.dataset.b);

                // 通过RGB值找到被点击的颜色对象
                const clickedColor = this.colors.find(c => c.r === r && c.g === g && c.b === b);
                if (!clickedColor) return; // 安全检查

                // 根据颜色在内部排好序的数组中的位置，确定其得分等级
                const rank = this.colors.indexOf(clickedColor);
                const basePoints = [6, 2, -2, -6][rank]; // 使用数组索引直接映射分数

                // 判断点击区域
                const rect = element.getBoundingClientRect();
                const clickX = event.clientX - rect.left;
                const clickY = event.clientY - rect.top;
                // 定义边缘区域为宽/高度的35%
                const edgeMargin = rect.width * 0.35;
                let positionPoints = 0;

                if (clickX > edgeMargin && clickX < rect.width - edgeMargin &&
                    clickY > edgeMargin && clickY < rect.height - edgeMargin) {
                    positionPoints = 1; // 点击中心区域 +1分
                } else {
                    positionPoints = -4; // 点击边缘区域 -4分
                }

                // 更新总分
                game.updateScore(basePoints + positionPoints);
            }
        },
        3: {
            rule: "单词的分数由其包含的元音字母决定（a:1, e:2, i:3, o:2, u:1），选择贬义词直接扣分，连续选择3次褒义词也会受到惩罚！",
            positiveStreak: 0,
            neutralStreak: 0,
            generate() {
                const gameContent = document.getElementById('game-content');
                const selectedWords = wordBank.sort(() => 0.5 - Math.random()).slice(0, 4);

                let html = '<p class="question">选择一个单词</p><div class="word-selection">';
                selectedWords.forEach(wordObj => {
                    html += `<button class="word-btn" onclick='game.levels[3].check(${JSON.stringify(wordObj)})'>${wordObj.word}</button>`;
                });
                html += '</div>';
                gameContent.innerHTML = html;
            },

            // 【修改点】重写 check 逻辑以处理三种词性
            check(wordObj) {
                let points = 0;
                // let feedbackMessage = '';

                // 提取元音计算为函数，避免重复
                const calculateVowelScore = (word) => {
                    const vowelScores = { 'a': 1, 'e': 2, 'i': 3, 'o': 2, 'u': 1 };
                    const uniqueVowels = [...new Set(word.match(/[aeiou]/g) || [])];
                    let score = 0;
                    uniqueVowels.forEach(vowel => { score += vowelScores[vowel]; });

                    return score;
                };

                switch (wordObj.type) {
                    case 'negative':
                        points = -8;
                        this.positiveStreak = 0; // 重置褒义词连击
                        this.neutralStreak = 0;  // 重置中性词连击
                        // feedbackMessage = `“${wordObj.word}” 是贬义词！连击已重置。`;
                        break;

                    case 'neutral':
                        points = calculateVowelScore(wordObj.word);
                        this.neutralStreak++; // 中性词连击+1

                        if (this.neutralStreak >= 2) {
                            this.positiveStreak = 0; // 达到2次，重置褒义词连击
                            this.neutralStreak = 0;
                            // feedbackMessage = `点击2次中性词！褒义词连击已重置。`;
                        } else {
                            // feedbackMessage = `“${wordObj.word}” 是中性词。（再点一次中性词将重置褒义词连击）`;
                        }
                        break;

                    case 'positive':
                        this.positiveStreak++; // 连击次数+1
                        if (this.positiveStreak >= 3) {
                            points = -12;
                            // feedbackMessage = "连续3次选择褒义词！触发惩罚！";
                            this.positiveStreak = 0; // 触发惩罚后，连击重置
                        } else {
                            points = calculateVowelScore(wordObj.word);
                            // feedbackMessage = `褒义词连击 x${this.positiveStreak}！`;
                        }
                        break;
                }

                // document.getElementById('feedback').innerText = ` ${feedbackMessage}`;
                game.updateScore(points);
            }
        }
    }
};

// 初始化游戏
document.addEventListener('DOMContentLoaded', () => {
    game.showMenu();
});