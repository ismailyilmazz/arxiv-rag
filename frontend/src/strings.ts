import type { DocType, JobStage, Lang } from "./api/types";

export const docTypes: Record<DocType, { name: string; description: string }> = {
  survey: {
    name: "Literatür taraması",
    description: "Alandaki yaklaşımları temalara ayırır, karşılaştırır ve açık problemleri çıkarır.",
  },
  proposal: {
    name: "Araştırma önerisi",
    description: "Literatürdeki boşluktan araştırma soruları, yöntem ve değerlendirme planı kurar.",
  },
  synthesis: {
    name: "Sentez raporu",
    description: "Kaynakların nerede uzlaştığını, nerede ayrıştığını ve bunun ne anlama geldiğini yazar.",
  },
};

export const languages: Record<Lang, string> = { tr: "Türkçe", en: "İngilizce" };

export const stages: Record<JobStage, string> = {
  sources: "Kaynaklar seçiliyor",
  cards: "Temel makaleler tam metinden okunuyor",
  sections: "Bölümler yazılıyor",
  translate: "Türkçeye çevriliyor",
  verify: "Atıflar doğrulanıyor",
  done: "Tamamlandı",
};

export const text = {
  title: "arXiv araştırma asistanı",
  topicQuestion: "Hangi konuda bir literatür çalışması istiyorsun?",
  topicPlaceholder: "Örneğin: büyük dil modellerinde uzman karışımı (mixture of experts)",
  topicLead:
    "Konunu yaz, ilgili arXiv makalelerini bulalım. Belgenin hangi kaynaklardan yazılacağını sen onaylarsın, sonra her iddiası kaynağına bağlı bir metin hazırlanır.",
  topicHint: "Türkçe ya da İngilizce yazabilirsin. Enter ile ara, Shift+Enter ile yeni satıra geç.",
  topicMissing: "Önce yukarıdaki kutuya bir konu yaz. Kutudaki gri yazı sadece bir örnek.",
  search: "Makaleleri bul",
  searching: "Aranıyor",
  steps: { topic: "Konu", plan: "Plan", document: "Belge" },
  documentType: "Belge türü",
  documentLanguage: "Belgenin dili",
  focus: "Özellikle ele alınacak yön (isteğe bağlı)",
  focusHint: "Belge konuyu bu açıdan işler. Boş bırakırsan konunun genelini ele alır.",
  focusPlaceholder: "Örneğin: düşük kaynaklı diller, çıkarım maliyeti",
  seedsLegend: "Konuyu en iyi temsil eden makaleleri işaretle",
  found: (count: number, picked: number) => `${count} makale bulundu, ${picked} tanesi seçili.`,
  seedsHint: "En fazla 5. İşaretlediklerin tam metinden okunur ve kaynak seçimine yön verir.",
  noResults: "Bu konuda makale bulunamadı. Konuyu daha genel ya da İngilizce terimlerle yaz.",
  weakMatch: "Bu konu korpustaki makalelerle zayıf eşleşiyor; sonuçlar ilgisiz olabilir.",
  forceAnyway: "Yine de plan oluştur",
  createPlan: "Plan oluştur",
  planHeading: (sources: number, themes: number) => `${sources} kaynak, ${themes} tema`,
  planHint:
    "Belgeye girmesini istemediğin kaynakların işaretini kaldır. Başlığın altındaki çizgi, makalenin konuya ne kadar yakın olduğunu gösterir.",
  showSources: "Kaynakları göster",
  edit: "Düzenle",
  deep: "Tam metinden okunacak",
  relevance: "Konuya yakınlık",
  furtherReading: "İleri okuma",
  furtherReadingHint: "Belgede atıflanmaz, kaynakçanın sonunda listelenir.",
  dropped: (n: number) => `Konu dışı bulunup çıkarılanlar (${n})`,
  rejectedQueries: "Şu alt sorgular korpusla eşleşmediği için atlandı:",
  writeDocument: "Belgeyi yaz",
  noSourcesLeft: "Yazmak için en az bir kaynak bırakmalısın.",
  queued: (position: number) => `Sırada, önünde ${position - 1} iş var`,
  waiting: "Başlatılıyor",
  documentSummary: (words: number, cited: number) => `${words} kelime, ${cited} atıflı kaynak`,
  downloadMarkdown: "Markdown olarak indir",
  downloadReport: "Üretim raporunu indir",
  newTopic: "Yeni konu",
  footer:
    "Makaleler ve linkler arXiv'den gelir. Metin bir dil modeliyle yazılır ve her atıf kaynak listesine karşı doğrulanır; önemli kararlar için kaynakların kendisini oku.",
  progress: "Adımlar",
  toasts: {
    found: (count: number) => `${count} makale bulundu. Plan oluşturmak için sonuçlara in.`,
    goResults: "Sonuçlara git",
    planAccepted: "Plan isteği alındı. Kaynaklar seçiliyor.",
    planReady: "Plan hazır.",
    goPlan: "Plana git",
    documentAccepted: "Belge isteği alındı. Yazım başladı.",
    documentReady: "Belge hazır.",
    goDocument: "Belgeye git",
    close: "Bildirimi kapat",
  },
  errors: {
    network: "Sunucuya ulaşılamadı. Bağlantını kontrol edip tekrar dene.",
    invalid: "Girdi kabul edilmedi. Konuyu kontrol edip tekrar dene.",
    unknown: "Beklenmeyen bir hata oluştu. Tekrar dene.",
  },
};
