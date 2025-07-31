import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('mobilenetv2_cat_breed_classification_model.keras')

# Daftar kelas
class_names = [
    'Abyssinian', 'Angora', 'Bengal', 'Domestic_shorthair', 'Maine_coon', 'Persian', 
    'Ragdoll', 'Scottish_Fold', 'Siamese', 'Sphynx', 'Tuxedo'
]

# Informasi tiap ras kucing
informasi_kucing = {
    "Anggora": {
        "penjelasan": "Kucing Anggora, atau dikenal juga sebagai Turkish Angora (Ankara kedisi dalam bahasa Turki), adalah salah satu ras kucing domestik alami tertua di dunia. Ras ini berasal dari Ankara, Turki, dan telah ada sejak zaman kekaisaran Romawi, bahkan dikenal sebagai simbol kemewahan. Anggora terkenal dengan bulunya yang panjang, indah, lembut, dan lebat, meskipun pada mulanya hanya Anggora putih murni yang diakui.",
        "ciri": "Berbulu panjang, tubuh ramping, ekor lebat, wajah runcing.",
        "perawatan": "Perawatan kucing Anggora memerlukan perhatian khusus, terutama pada bulunya yang panjang dan indah. Sikat bulu mereka secara rutin, setidaknya beberapa kali seminggu, untuk mencegah kusut dan menghilangkan bulu mati. Mandikan Anggora secara teratur menggunakan sampo khusus kucing, pastikan bulunya kering sempurna. Berikan makanan berkualitas tinggi yang kaya nutrisi untuk menjaga kesehatan bulu dan tubuhnya. Jangan lupa untuk rutin membersihkan telinga, memotong kuku, dan menyikat gigi mereka. Karena Anggora adalah kucing yang aktif dan cerdas, sediakan banyak waktu untuk bermain dan berikan stimulasi mental serta fisik yang cukup. Terakhir, lakukan pemeriksaan kesehatan rutin ke dokter hewan dan berikan kasih sayang serta perhatian agar Anggora Anda selalu sehat dan bahagia.",
        "gambar": "gambar/angora.jpg"
    },
    "Persia": {
        "penjelasan": "Kucing Persia adalah salah satu ras kucing berbulu panjang yang paling populer di dunia, dikenal luas karena penampilannya yang mewah dan ekspresi wajahnya yang unik. Berasal dari Persia (sekarang Iran), ras ini telah dikembangkan selama berabad-abad hingga memiliki ciri khas yang kita kenal sekarang. Persia ideal untuk orang yang mencari kucing pendamping yang tenang dan penyayang, karena mereka cenderung tidak terlalu aktif dibanding ras lain seperti Anggora.",
        "ciri": "Wajah bulat, hidung pesek, berbulu tebal dan panjang.",
        "perawatan": "Perawatan kucing Persia membutuhkan komitmen tinggi, terutama karena bulunya yang lebat dan wajahnya yang pesek. Sikat bulunya setiap hari menggunakan sisir khusus kucing berbulu panjang untuk mencegah gumpalan bulu, menghilangkan bulu mati, dan menjaga keindahan serta kelembutannya; jika ada bulu gimbal, segera potong. Mandikan secara teratur, setidaknya sebulan sekali, dengan sampo khusus kucing, pastikan bulu benar-benar kering untuk menghindari masalah kulit dan jamur. Karena wajahnya yang pesek, bersihkan area mata dan hidung setiap hari dengan lembut menggunakan kapas basah untuk menghilangkan kotoran dan noda air mata yang sering menumpuk. Selain itu, potong kuku secara rutin, bersihkan telinga, dan sikat giginya untuk menjaga kebersihan dan kesehatan umum. Berikan makanan berkualitas tinggi yang diformulasikan untuk kucing berbulu panjang untuk mendukung kesehatan kulit dan bulu. Jangan lupakan pemeriksaan rutin ke dokter hewan dan berikan banyak cinta serta perhatian agar kucing Persia Anda tetap bahagia dan sehat.",
        "gambar": "gambar/persian.jpg"
    },
    "Maine Coon": {
        "penjelasan": "Maine Coon adalah salah satu ras kucing domestik terbesar dan tertua di Amerika Utara, khususnya berasal dari negara bagian Maine. Mereka sering dijuluki -raksasa lembut- karena ukurannya yang besar namun temperamennya sangat ramah dan sabar. Maine Coon dikenal sebagai kucing yang cerdas, setia, dan mudah dilatih, serta sering menunjukkan perilaku mirip anjing, seperti mengikuti pemiliknya atau suka diajak berjalan-jalan dengan tali.",
        "ciri": "Tubuh besar, bulu lebat terutama di leher, telinga berjumbai.",
        "perawatan": "Perawatan kucing Maine Coon relatif mudah dibandingkan ras berbulu panjang lainnya. Sikat bulunya 2-3 kali seminggu untuk mencegah kusut dan menghilangkan bulu mati, terutama bagian perut. Meskipun bulunya tahan air, mandikan mereka secara teratur (setiap 1-2 bulan atau sesuai kebutuhan) untuk menjaga kebersihan dan kesehatan kulit. Berikan makanan kucing berkualitas tinggi yang diformulasikan untuk kucing berukuran besar atau aktif untuk mendukung kebutuhan energinya. Rutin bersihkan telinga, potong kuku, dan sikat gigi. Karena sifatnya yang aktif dan cerdas, sediakan banyak mainan interaktif, pohon kucing untuk memanjat, dan waktu bermain untuk stimulasi fisik dan mental. Jangan lupakan pemeriksaan kesehatan rutin ke dokter hewan untuk memastikan mereka tumbuh sehat dan bebas dari masalah genetik tertentu seperti kardiomiopati hipertrofik atau displasia pinggul.",
        "gambar": "gambar/mainecoon.jpg"
    },
    "Bengal": {
        "penjelasan": "Kucing Bengal adalah ras hibrida yang menarik, hasil persilangan antara kucing domestik dengan Asian Leopard Cat (kucing macan tutul Asia). Mereka dikembangkan untuk memiliki penampilan eksotis seperti kucing liar namun dengan temperamen kucing peliharaan. Bengal dikenal dengan pola bulu tutul atau marble yang unik dan kepribadiannya yang aktif serta penasaran.",
        "ciri": "Bulu bermotif tutul/loreng seperti macan, tubuh atletis.",
        "perawatan": "Perawatan kucing Bengal relatif mudah karena bulunya yang pendek dan mudah dirawat. Sikat bulunya seminggu sekali untuk menghilangkan bulu mati dan menjaga kilaunya. Mandikan mereka hanya jika benar-benar kotor, karena mereka umumnya pandai merawat diri. Karena tingkat energinya yang tinggi, sediakan banyak mainan interaktif, tiang garuk, dan pohon kucing untuk memanjat. Luangkan waktu bermain setiap hari untuk menyalurkan energinya dan mencegah kebosanan yang bisa menyebabkan perilaku destruktif. Pastikan mereka memiliki akses ke tempat yang tinggi, karena mereka suka mengamati dari ketinggian. Berikan makanan berkualitas tinggi yang mendukung tingkat aktivitas mereka yang tinggi. Jangan lupa rutin membersihkan telinga, memotong kuku, dan sikat gigi. Kunjungan dokter hewan secara teratur penting untuk memantau kesehatan dan memastikan vaksinasi lengkap.",
        "gambar": "gambar/bengal.jpg"
    },
    "Abyssinian": {
        "penjelasan": "Abyssinian adalah salah satu ras kucing tertua, meskipun asal-usul pastinya masih diperdebatkan (nama Abyssinian merujuk pada Ethiopia modern). Mereka dikenal dengan penampilannya yang elegan, atletis, dan bulu ticked yang unik, serta kepribadiannya yang ceria dan ingin tahu. Abyssinian adalah kucing yang sangat interaktif dan suka menjadi pusat perhatian.",
        "ciri": "Bulu pendek, ramping, telinga besar, aktif dan cerdas.",
        "perawatan": "Perawatan kucing Abyssinian terbilang minimal karena bulunya yang pendek. Sikat bulunya seminggu sekali untuk menghilangkan bulu mati dan menjaga kilaunya. Mandikan mereka hanya jika diperlukan, karena mereka sangat bersih. Karena sifatnya yang sangat aktif dan ingin tahu, penting untuk menyediakan banyak stimulasi mental dan fisik. Berikan berbagai mainan interaktif, teka-teki makanan, dan pohon kucing untuk memanjat. Mereka juga sangat menyukai interaksi dengan manusia, jadi luangkan banyak waktu bermain dan berinteraksi dengan mereka setiap hari. Potong kuku, bersihkan telinga, dan sikat gigi secara rutin. Berikan makanan kucing berkualitas tinggi yang sesuai dengan tingkat aktivitasnya. Pemeriksaan kesehatan rutin ke dokter hewan juga penting untuk memastikan mereka tetap sehat dan mendeteksi dini masalah genetik tertentu yang mungkin mereka miliki.",
        "gambar": "gambar/abyssinian.jpg"
    },
    "Domestic Shorthair": {
        "penjelasan": "Domestic Shorthair (DSH) sebenarnya bukan ras kucing yang diakui secara formal, melainkan istilah umum untuk kucing domestik campuran yang memiliki bulu pendek. Mereka adalah kucing yang paling umum ditemukan di seluruh dunia dan sering disebut sebagai kucing kampung atau kucing rumahan. Karena mereka bukan hasil pembiakan selektif, DSH memiliki keragaman genetik yang luas, membuat mereka sangat tangguh, sehat, dan adaptif. Mereka mewarisi berbagai ciri fisik dan kepribadian dari nenek moyang mereka.",
        "ciri": "Campuran berbagai ras, bulu pendek, sangat umum dan beragam warna.",
        "perawatan": "Perawatan kucing Domestic Shorthair sangat mudah karena bulunya yang pendek dan minim rontok. Sikat bulunya seminggu sekali sudah cukup untuk menghilangkan bulu mati dan menjaga kebersihannya. Mandikan hanya jika benar-benar kotor, karena mereka umumnya pandai menjaga diri. Berikan makanan kucing berkualitas baik yang sesuai dengan usia dan tingkat aktivitasnya. Rutin potong kuku, bersihkan telinga, dan sikat gigi untuk menjaga kebersihan dan mencegah masalah kesehatan mulut. Meskipun tidak memerlukan stimulasi khusus seperti ras lain, mereka tetap membutuhkan waktu bermain dan interaksi untuk menjaga kebahagiaan dan mencegah kebosanan. Jangan lupa pemeriksaan rutin ke dokter hewan dan vaksinasi yang diperlukan untuk memastikan mereka tetap sehat dan panjang umur.",
        "gambar": "gambar/domestic_shorthair.jpg"
    },
    "Ragdoll": {
        "penjelasan": "Ragdoll adalah ras kucing berbulu semi-panjang yang terkenal dengan temperamennya yang sangat lembut, tenang, dan sifatnya yang lemas saat diangkat – mirip boneka kain (rag doll), dari sinilah namanya berasal. Mereka adalah kucing yang sangat penyayang dan setia, sering mengikuti pemiliknya seperti anak anjing dan menyukai interaksi. Ragdoll cocok sebagai hewan peliharaan keluarga karena kesabarannya yang tinggi.",
        "ciri": "Tubuh besar, mata biru, bulu panjang, jinak dan kalem.",
        "perawatan": "Perawatan kucing Ragdoll relatif mudah meskipun bulunya semi-panjang. Sikat bulunya 2-3 kali seminggu untuk mencegah gumpalan dan menghilangkan bulu mati, namun karena bulunya tidak mudah kusut, ini tidak terlalu memakan waktu. Mandikan mereka setiap 1-2 bulan sekali atau sesuai kebutuhan untuk menjaga kebersihan dan kilau bulu. Berikan makanan berkualitas tinggi yang mendukung ukuran tubuh besar mereka. Rutin potong kuku, bersihkan telinga, dan sikat gigi adalah bagian penting dari rutinitas perawatan. Karena sifatnya yang sangat sosial dan penyayang, sediakan banyak waktu untuk berinteraksi, bermain, dan memeluk mereka. Mereka tidak terlalu aktif melompat, jadi mainan lantai dan tempat tidur yang nyaman sudah cukup. Jangan lupakan pemeriksaan kesehatan rutin ke dokter hewan untuk memantau kesehatan dan memastikan vaksinasi lengkap.",
        "gambar": "gambar/ragdoll.jpg"
    },
    "Scottish Fold": {
        "penjelasan": "Scottish Fold adalah ras kucing yang sangat unik dan mudah dikenali berkat telinganya yang melipat ke depan dan ke bawah, memberikan tampilan seperti burung hantu atau teddy bear. Ciri telinga ini disebabkan oleh mutasi genetik dominan alami. Mereka berasal dari Skotlandia dan dikenal dengan temperamennya yang manis, tenang, dan mudah beradaptasi, menjadikannya hewan peliharaan keluarga yang populer.",
        "ciri": "Telinga terlipat ke bawah, wajah bulat.",
        "perawatan": "Perawatan kucing Scottish Fold memerlukan perhatian khusus pada telinganya. Periksa dan bersihkan telinganya secara rutin (mingguan) dengan lembut menggunakan kapas basah atau pembersih telinga khusus kucing, karena bentuk lipatannya bisa membuat telinga lebih rentan terhadap penumpukan kotoran dan infeksi. Untuk Scottish Fold berbulu pendek, sikat bulunya seminggu sekali, sementara untuk yang berbulu panjang, sikat 2-3 kali seminggu. Mandikan hanya jika diperlukan. Berikan makanan berkualitas tinggi untuk menjaga kesehatan umum. Rutin potong kuku dan sikat gigi. Karena mereka cenderung tidak seaktif ras lain, sediakan mainan interaktif dan waktu bermain yang cukup untuk menjaga mereka tetap aktif. Penting untuk melakukan pemeriksaan kesehatan rutin ke dokter hewan, karena mutasi genetik yang menyebabkan telinga melipat juga dapat memengaruhi tulang rawan di bagian tubuh lain, menyebabkan kondisi yang disebut osteokondrodisplasia yang bisa menyebabkan radang sendi dan rasa sakit. Pemilik harus memantau tanda-tanda ketidaknyamanan atau kekakuan pada sendi.",
        "gambar": "gambar/scottish_fold.jpg"
    },
    "Siamese": {
        "penjelasan": "Kucing Siamese adalah salah satu ras kucing berbulu pendek paling ikonik dan tertua di dunia, berasal dari Siam (sekarang Thailand). Mereka dikenal dengan ciri khas colorpoint (warna lebih gelap di bagian tubuh tertentu), mata biru yang menusuk, dan kepribadiannya yang sangat vokal serta cerdas. Siamese adalah kucing yang sangat sosial dan membentuk ikatan kuat dengan pemiliknya, sering mengikuti ke mana pun mereka pergi.",
        "ciri": "Tubuh ramping, mata biru terang, bulu pendek, suara nyaring.",
        "perawatan": "Perawatan kucing Siamese cukup mudah karena bulunya yang pendek dan minim rontok. Sikat bulunya seminggu sekali untuk menghilangkan bulu mati dan menjaga kilau bulunya. Mandikan hanya jika benar-benar kotor, karena mereka umumnya pandai menjaga diri. Berikan makanan kucing berkualitas tinggi yang sesuai dengan tingkat energi dan usia mereka. Rutin potong kuku, bersihkan telinga, dan sikat gigi untuk menjaga kebersihan dan kesehatan mulut. Karena sifatnya yang sangat sosial, cerdas, dan aktif, sediakan banyak mainan interaktif, teka-teki makanan, dan waktu bermain setiap hari untuk menstimulasi pikiran dan tubuh mereka. Mereka tidak suka ditinggal sendirian terlalu lama, jadi pastikan mereka mendapat banyak perhatian. Pemeriksaan kesehatan rutin ke dokter hewan juga penting untuk memantau kesehatan dan memastikan vaksinasi lengkap.",
        "gambar": "gambar/siamese.jpg"
    },
    "Sphynx": {
        "penjelasan": "Kucing Sphynx adalah ras kucing yang sangat unik dan mudah dikenali karena kekurangan bulu mereka. Meskipun sering disebut tidak berbulu, mereka sebenarnya ditutupi oleh lapisan bulu halus seperti persik yang nyaris tidak terlihat. Ras ini dikembangkan melalui pembiakan selektif di Kanada. Sphynx dikenal dengan kulitnya yang keriput, telinga besar, dan kepribadiannya yang sangat ramah, penyayang, dan suka mencari perhatian.",
        "ciri": "Tidak berbulu, kulit berkerut, tubuh hangat dan aktif.",
        "perawatan": "Perawatan kucing Sphynx sangat berbeda dari kucing berbulu. Mandikan mereka secara rutin (sekali seminggu) menggunakan sampo khusus kucing, karena minyak alami kulit mereka tidak diserap oleh bulu dan bisa menumpuk, menyebabkan kulit berminyak atau berjerawat. Setelah mandi, pastikan kulit mereka benar-benar kering. Bersihkan lipatan kulitnya secara teratur, terutama di sekitar wajah, leher, dan di antara jari kaki, untuk mencegah penumpukan kotoran atau infeksi. Karena tidak memiliki bulu, Sphynx rentan terhadap perubahan suhu; lindungi mereka dari cuaca dingin dengan pakaian kucing atau selimut, dan dari paparan sinar matahari langsung untuk mencegah kulit terbakar. Berikan makanan berkualitas tinggi yang mendukung kebutuhan energi mereka. Rutin bersihkan telinga mereka secara ekstra hati-hati karena kotoran telinga lebih terlihat, potong kuku, dan sikat gigi. Karena sifatnya yang sosial, berikan banyak perhatian dan waktu bermain setiap hari. Pemeriksaan kesehatan rutin ke dokter hewan sangat penting untuk memantau kesehatan mereka, karena mereka bisa rentan terhadap masalah kulit dan jantung tertentu.",
        "gambar": "gambar/sphynx.jpg"
    },
    "Tuxedo": {
        "penjelasan": "Sama seperti Domestic Shorthair, Tuxedo bukanlah ras kucing yang diakui secara formal, melainkan adalah istilah yang digunakan untuk menggambarkan kucing dengan pola bulu hitam solid dengan bercak putih di dada, perut, cakar (seperti sarung tangan), dan terkadang wajah (seperti dasi kupu-kupu). Penampilannya yang rapi menyerupai setelan tuksedo, dari situlah namanya berasal. Kucing Tuxedo bisa berbulu pendek maupun panjang, meskipun yang berbulu pendek lebih umum. Karakteristik kepribadian dan fisik mereka sangat bervariasi karena mereka adalah kucing campuran.",
        "ciri": "Pola warna hitam-putih seperti jas formal.",
        "perawatan": "Ikuti standar perawatan Domestic Shorthair: mudah dirawat, cukup disikat dan diberi makanan berkualitas.",
        "gambar": "gambar/tuxedo.jpg"
    }
}

# Judul utama
st.title("Klasifikasi Ras Kucing dengan CNN")

# Menampilkan informasi semua kelas dengan expander
st.subheader("🐾 Informasi Ras Kucing")
for ras, info in informasi_kucing.items():
    with st.expander(f"ℹ️ {ras}"):
        st.image(info["gambar"], use_container_width=True)
        st.markdown(f"**Penjelasan:** {info['penjelasan']}")
        st.markdown(f"**Ciri-ciri:** {info['ciri']}")
        st.markdown(f"**Perawatan:** {info['perawatan']}")

# Upload untuk klasifikasi
st.subheader("🔍 Prediksi Ras Kucing dari Gambar")

uploaded_file = st.file_uploader("Upload gambar kucing", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")  # Konversi ke RGB
    st.image(img, caption="Gambar yang di-upload", use_container_width=True)

    # Preprocessing
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi dan ambil confidence
    predictions = model.predict(img_array)
    prediction = predictions[0]
    confidence = np.max(prediction)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence_percent = confidence * 100

    # Tampilkan hasil
    if confidence_percent >= 50:
        st.success(f"🎉 Prediksi: **{predicted_class}**")
        st.info(f"📊 Akurasi: {confidence_percent:.2f}%")
    else:
        st.error("🚫 Ini kemungkinan **bukan kucing** atau gambar tidak jelas.")
        st.info(f"📊 Keyakinan model terlalu rendah: hanya {confidence_percent:.2f}%")