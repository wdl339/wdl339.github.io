game.levels[1]  = {
    rule: "颜色的隐藏分数由它的 RGB 值的 G-R+B 计算，值越大奖励越大。点击色块中心有额外加分，边缘区域会减分。",
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
                        onclick="game.levels[1].check(this, event)"></div>`;
        });
        html += '</div>';
        html += `<div class="back-btn-container">
                    <button onclick="game.showMenu()">返回菜单</button>
                </div>`;
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
        // 定义边缘区域为宽/高度的40%
        const edgeMargin = rect.width * 0.4;
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
}