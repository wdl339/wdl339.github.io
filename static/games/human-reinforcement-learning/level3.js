game.levels[3]  = {
    rule: "单词的分数由其包含的元音字母决定（a:1, e:2, i:3, o:2, u:1），选择贬义词直接扣分，连续选择3次褒义词也会受到惩罚！",
    positiveStreak: 0,
    neutralStreak: 0,

    generate() {
        const gameContent = document.getElementById('game-content');
        const selectedWords = wordBank.sort(() => 0.5 - Math.random()).slice(0, 4);

        let html = '<p class="question">选择一个单词</p><div id="main-menu"><div class="button-container">';
        selectedWords.forEach(wordObj => {
            html += `<button class="word-btn" onclick='game.levels[3].check(${JSON.stringify(wordObj)})'>${wordObj.word}</button>`;
        });
        html += '</div></div>';
        html += `<div class="back-btn-container">
                    <button onclick="game.showMenu()">返回菜单</button>
                </div>`;
        gameContent.innerHTML = html;
    },

    check(wordObj) {
        let points = 0;

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
                points = calculateVowelScore(wordObj.word);
                points -= 12;
                this.positiveStreak = 0; // 重置褒义词连击
                this.neutralStreak = 0;  // 重置中性词连击
                break;

            case 'neutral':
                points = calculateVowelScore(wordObj.word);
                this.neutralStreak++; // 中性词连击+1

                if (this.neutralStreak >= 2) {
                    this.positiveStreak = 0; // 达到2次，重置褒义词连击
                    this.neutralStreak = 0;
                }
                break;

            case 'positive':
                this.positiveStreak++; // 连击次数+1
                if (this.positiveStreak >= 3) {
                    points = -12;
                    this.positiveStreak = 0; // 触发惩罚后，连击重置
                } else {
                    points = calculateVowelScore(wordObj.word);
                }
                break;
        }

        game.updateScore(points);
    }
}