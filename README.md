# لغة الإشارة اليمنية - Yemeni Sign Language

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Platform](https://img.shields.io/badge/platform-Web%20%7C%20Mobile-green)
![Language](https://img.shields.io/badge/language-Arabic%20%7C%20English-orange)

تطبيق ويب لترجمة وتسجيل لغة الإشارة اليمنية باستخدام الذكاء الاصطناعي

**A web application for translating and recording Yemeni Sign Language using AI**

</div>

---

## 🌟 المميزات | Features

### 🧏 وضع المستخدم الأصم | Deaf User Mode
- ترجمة النص العربي إلى لغة الإشارة
- عرض الإشارات عبر شخصية ثلاثية الأبعاد (Avatar)
- دعم الإدخال الصوتي

### 👂 وضع المستخدم السامع | Hearing User Mode
- التعرف على إشارات اليد عبر الكاميرا
- ترجمة الإشارات إلى نص عربي
- دعم النطق الصوتي

### 📹 وضع التسجيل | Recording Mode
- تسجيل حركات الإشارة الجديدة
- تتبع اليدين والجسم والوجه
- حفظ الإشارات في القاموس

---

## 🚀 التثبيت | Installation

### المتطلبات | Requirements
- Node.js (v16 أو أحدث)
- متصفح حديث (Chrome, Firefox, Edge)

### خطوات التثبيت | Steps

```bash
# 1. استنساخ المشروع | Clone the project
git clone https://github.com/YOUR_USERNAME/yemeni-sign-language.git

# 2. الدخول للمجلد | Enter directory
cd yemeni-sign-language

# 3. تثبيت الحزم | Install packages
npm install

# 4. تشغيل الخادم | Start server
npm start
```

ثم افتح المتصفح على: `http://localhost:8080`

---

## 📁 هيكل المشروع | Project Structure

```
yemeni-sign-language/
├── index.js              # الخادم الرئيسي | Main server
├── package.json          # إعدادات المشروع | Project config
├── public/
│   ├── welcome.html      # الصفحة الرئيسية | Home page
│   ├── translator.html   # صفحة المترجم | Translator page
│   ├── recognizer.html   # صفحة التعرف | Recognizer page
│   ├── recording.html    # صفحة التسجيل | Recording page
│   ├── dictionary.json   # قاموس الكلمات | Word dictionary
│   ├── animations/       # ملفات الإشارات | Animation files
│   └── src/              # ملفات JavaScript
└── README.md
```

---

## 📖 كيفية الاستخدام | How to Use

### للمستخدم الأصم | For Deaf Users
1. افتح صفحة "المترجم" (Translator)
2. اكتب النص بالعربية
3. اضغط "ترجم وشغل"
4. شاهد الشخصية تؤدي الإشارات

### للمستخدم السامع | For Hearing Users
1. افتح صفحة "المتعرف" (Recognizer)
2. اسمح بالوصول للكاميرا
3. قم بأداء الإشارة أمام الكاميرا
4. سترى الترجمة على الشاشة

### لتسجيل إشارة جديدة | To Record New Sign
1. افتح صفحة "التسجيل" (Recording)
2. اضغط على زر الكاميرا
3. اختر "يد واحدة" أو "يدين"
4. اضغط "ابدأ التسجيل" وأدِّ الإشارة
5. اضغط "إيقاف" ثم "حفظ"

---

## 🌐 النشر | Deployment

### Render.com (مجاني | Free)
1. ارفع المشروع إلى GitHub
2. اذهب إلى [render.com](https://render.com)
3. أنشئ Web Service جديد
4. اربطه بمستودع GitHub
5. Build Command: `npm install`
6. Start Command: `npm start`

---

## 📝 القاموس | Dictionary

الكلمات المتاحة حالياً:
| العربية | English |
|---------|---------|
| الأم | MOTHER |
| الأب | FATHER |
| نعم | YES |
| لا | NO |
| شكرا | THANKYOU |
| ... | ... |

---

## 🤝 المساهمة | Contributing

نرحب بمساهماتكم! يرجى فتح Issue أو Pull Request.

## 📄 الترخيص | License

MIT License

---

<div align="center">

صُنع بـ ❤️ في اليمن | Made with ❤️ in Yemen

</div>
