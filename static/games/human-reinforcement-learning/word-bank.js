const wordBank = [
    // --- 褒义词 (Positive) ---
    { word: 'love', type: 'positive' }, { word: 'joy', type: 'positive' },
    { word: 'happy', type: 'positive' }, { word: 'friend', type: 'positive' },
    { word: 'dream', type: 'positive' }, { word: 'success', type: 'positive' },
    { word: 'great', type: 'positive' }, { word: 'amazing', type: 'positive' },
    { word: 'victory', type: 'positive' }, { word: 'genius', type: 'positive' },
    { word: 'spirit', type: 'positive' }, { word: 'honor', type: 'positive' },
    { word: 'glory', type: 'positive' }, { word: 'kind', type: 'positive' },
    { word: 'bliss', type: 'positive' }, // 极乐，狂喜
    { word: 'serene', type: 'positive' }, // 安详的，宁静的
    { word: 'vibrant', type: 'positive' }, // 充满活力的
    { word: 'brave', type: 'positive' }, { word: 'award', type: 'positive' },
    { word: 'faith', type: 'positive' }, { word: 'merit', type: 'positive' },
    { word: 'proud', type: 'positive' }, { word: 'sweet', type: 'positive' },
    { word: 'thrive', type: 'positive' }, // 茁壮成长
    { word: 'wisdom', type: 'positive' }, { word: 'unity', type: 'positive' },
    { word: 'bonus', type: 'positive' }, { word: 'asset', type: 'positive' },
    { word: 'hero', type: 'positive' }, { word: 'angel', type: 'positive' },
    { word: 'loyal', type: 'positive' }, { word: 'pure', type: 'positive' },

    // --- 贬义词 (Negative) ---
    { word: 'hate', type: 'negative' }, { word: 'sad', type: 'negative' },
    { word: 'ugly', type: 'negative' }, { word: 'pain', type: 'negative' },
    { word: 'fear', type: 'negative' }, { word: 'angry', type: 'negative' },
    { word: 'stupid', type: 'negative' }, { word: 'evil', type: 'negative' },
    { word: 'loser', type: 'negative' }, { word: 'boring', type: 'negative' },
    { word: 'stress', type: 'negative' }, { word: 'gloom', type: 'negative' }, // 忧郁
    { word: 'dread', type: 'negative' }, // 恐惧
    { word: 'malice', type: 'negative' }, // 恶意
    { word: 'wreck', type: 'negative' }, // 残骸，毁坏
    { word: 'brute', type: 'negative' }, // 残忍的人
    { word: 'chaos', type: 'negative' }, { word: 'demon', type: 'negative' },
    { word: 'fault', type: 'negative' }, { word: 'guilt', type: 'negative' },
    { word: 'abyss', type: 'negative' }, // 深渊
    { word: 'curse', type: 'negative' }, { word: 'grief', type: 'negative' },
    { word: 'panic', type: 'negative' }, { word: 'smog', type: 'negative' },
    { word: 'toxic', type: 'negative' }, { word: 'trap', type: 'negative' },
    { word: 'void', type: 'negative' }, // 空虚，虚空
    { word: 'zero', type: 'negative' }, { word: 'phobia', type: 'negative' }, // 恐惧症

    // --- 中性词 (Neutral) ---
    { word: 'table', type: 'neutral' }, { word: 'water', type: 'neutral' },
    { word: 'book', type: 'neutral' }, { word: 'system', type: 'neutral' },
    { word: 'road', type: 'neutral' }, { word: 'air', type: 'neutral' },
    { word: 'box', type: 'neutral' }, { word: 'story', type: 'neutral' },
    { word: 'index', type: 'neutral' }, { word: 'pixel', type: 'neutral' },
    { word: 'epoch', type: 'neutral' }, // 纪元，时代
    { word: 'proxy', type: 'neutral' }, // 代理
    { word: 'glyph', type: 'neutral' }, // 象形文字，符号
    { word: 'rhythm', type: 'neutral' }, // 节奏（没有元音字母）
    { word: 'myth', type: 'neutral' }, // 神话
    { word: 'nymph', type: 'neutral' }, // 宁芙（希腊神话中的女神）
    { word: 'flask', type: 'neutral' }, // 烧瓶
    { word: 'cloth', type: 'neutral' }, { word: 'steel', type: 'neutral' },
    { word: 'wood', type: 'neutral' }, { word: 'stone', type: 'neutral' },
    { word: 'data', type: 'neutral' }, { word: 'fact', type: 'neutral' },
    { word: 'item', type: 'neutral' }, { word: 'list', type: 'neutral' },
    { word: 'note', type: 'neutral' }, { word: 'part', type: 'neutral' },
    { word: 'plan', type: 'neutral' }, { word: 'print', type: 'neutral' },
    { word: 'proof', type: 'neutral' }, { word: 'query', type: 'neutral' },
    { word: 'scope', type: 'neutral' }, { word: 'shell', type: 'neutral' },
    { word: 'stack', type: 'neutral' }, { word: 'state', type: 'neutral' },
    { word: 'steam', type: 'neutral' }, { word: 'task', type: 'neutral' },
    { word: 'text', type: 'neutral' }, { word: 'theme', type: 'neutral' },
    { word: 'thread', type: 'neutral' }
];