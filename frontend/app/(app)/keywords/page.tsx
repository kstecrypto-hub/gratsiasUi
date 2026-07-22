"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { ErrorState, LoadingState, TableEmpty } from "@/components/page-state";
import { SortButton } from "@/components/sort-button";
import { StatusLabel } from "@/components/status-label";
import { api, asList, messageFromError } from "@/lib/api";
import type { Identifier, Keyword, KeywordCategory } from "@/lib/types";

type KeywordDraft = {
  categoryId: string;
  phrase: string;
  variants: string;
  accentInsensitive: boolean;
  wholeWord: boolean;
  exactPhrase: boolean;
  fuzzyMatch: boolean;
  fuzzyThreshold: number;
  active: boolean;
  severity: string;
  notes: string;
};

const emptyKeyword: KeywordDraft = {
  categoryId: "",
  phrase: "",
  variants: "",
  accentInsensitive: true,
  wholeWord: true,
  exactPhrase: true,
  fuzzyMatch: false,
  fuzzyThreshold: 0.85,
  active: true,
  severity: "medium",
  notes: "",
};

export default function KeywordsPage() {
  const [categories, setCategories] = useState<KeywordCategory[]>([]);
  const [keywords, setKeywords] = useState<Keyword[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  const [success, setSuccess] = useState("");
  const [saving, setSaving] = useState(false);
  const [categoryName, setCategoryName] = useState("");
  const [categoryDescription, setCategoryDescription] = useState("");
  const [editingCategory, setEditingCategory] = useState<Identifier>();
  const [draft, setDraft] = useState<KeywordDraft>(emptyKeyword);
  const [editingKeyword, setEditingKeyword] = useState<Identifier>();
  const [sortColumn, setSortColumn] = useState("canonical_phrase");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [categoryPayload, keywordPayload] = await Promise.all([api.categories.list(), api.keywords.list()]);
      const nextCategories = asList(categoryPayload);
      setCategories(nextCategories);
      setKeywords(asList(keywordPayload));
      setDraft((current) => ({ ...current, categoryId: current.categoryId || (nextCategories[0] ? String(nextCategories[0].id) : "") }));
    } catch (caught) {
      setError(messageFromError(caught));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const sortedKeywords = useMemo(() => [...keywords].sort((a, b) => {
    const categoryName = (keyword: Keyword) => keyword.category_name || categories.find((category) => String(category.id) === String(keyword.category_id))?.name || "";
    const left = sortColumn === "category" ? categoryName(a) : sortColumn === "active" ? String(a.active) : a.canonical_phrase;
    const right = sortColumn === "category" ? categoryName(b) : sortColumn === "active" ? String(b.active) : b.canonical_phrase;
    return left.localeCompare(right, undefined, { sensitivity: "base" }) * (sortOrder === "asc" ? 1 : -1);
  }), [categories, keywords, sortColumn, sortOrder]);

  function sort(column: string) {
    if (sortColumn === column) setSortOrder((current) => current === "asc" ? "desc" : "asc");
    else { setSortColumn(column); setSortOrder("asc"); }
  }

  function beginCategoryEdit(category: KeywordCategory) {
    setEditingCategory(category.id);
    setCategoryName(category.name);
    setCategoryDescription(category.description || "");
    setActionError("");
  }

  function resetCategory() {
    setEditingCategory(undefined);
    setCategoryName("");
    setCategoryDescription("");
  }

  async function saveCategory(event: FormEvent) {
    event.preventDefault();
    if (!categoryName.trim()) return;
    setSaving(true);
    setActionError("");
    setSuccess("");
    try {
      if (editingCategory !== undefined) await api.categories.update(editingCategory, { name: categoryName.trim(), description: categoryDescription.trim() || null });
      else await api.categories.create({ name: categoryName.trim(), description: categoryDescription.trim() || null });
      setSuccess(editingCategory !== undefined ? "Category updated." : "Category created.");
      resetCategory();
      await load();
    } catch (caught) {
      setActionError(messageFromError(caught, "The category could not be saved."));
    } finally { setSaving(false); }
  }

  async function deleteCategory(category: KeywordCategory) {
    if (!window.confirm(`Delete the category “${category.name}”? This cannot be undone.`)) return;
    setActionError("");
    setSuccess("");
    try {
      await api.categories.remove(category.id);
      setSuccess("Category deleted.");
      if (editingCategory !== undefined && String(editingCategory) === String(category.id)) resetCategory();
      await load();
    } catch (caught) { setActionError(messageFromError(caught, "The category could not be deleted.")); }
  }

  function beginKeywordEdit(keyword: Keyword) {
    setEditingKeyword(keyword.id);
    setDraft({
      categoryId: String(keyword.category_id),
      phrase: keyword.canonical_phrase,
      variants: (keyword.variants || []).map((variant) => typeof variant === "string" ? variant : variant.phrase).join("\n"),
      accentInsensitive: keyword.accent_insensitive,
      wholeWord: keyword.whole_word,
      exactPhrase: keyword.exact_phrase,
      fuzzyMatch: keyword.fuzzy_match,
      fuzzyThreshold: keyword.fuzzy_threshold ?? 0.85,
      active: keyword.active,
      severity: keyword.severity,
      notes: keyword.notes || "",
    });
    document.getElementById("keyword-editor")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function resetKeyword() {
    setEditingKeyword(undefined);
    setDraft({ ...emptyKeyword, categoryId: categories[0] ? String(categories[0].id) : "" });
  }

  async function saveKeyword(event: FormEvent) {
    event.preventDefault();
    if (!draft.categoryId || !draft.phrase.trim()) return;
    setSaving(true);
    setActionError("");
    setSuccess("");
    const input = {
      category_id: draft.categoryId,
      canonical_phrase: draft.phrase.trim(),
      variants: draft.variants.split(/\r?\n/).map((phrase) => phrase.trim()).filter(Boolean).map((phrase) => ({ phrase })),
      accent_insensitive: draft.accentInsensitive,
      whole_word: draft.wholeWord,
      exact_phrase: draft.exactPhrase,
      fuzzy_match: draft.fuzzyMatch,
      fuzzy_threshold: draft.fuzzyThreshold,
      active: draft.active,
      severity: draft.severity,
      notes: draft.notes.trim() || null,
    };
    try {
      if (editingKeyword !== undefined) await api.keywords.update(editingKeyword, input);
      else await api.keywords.create(input);
      setSuccess(editingKeyword !== undefined ? "Phrase updated." : "Phrase added.");
      resetKeyword();
      await load();
    } catch (caught) { setActionError(messageFromError(caught, "The phrase could not be saved.")); }
    finally { setSaving(false); }
  }

  async function toggleKeyword(keyword: Keyword) {
    setActionError("");
    try { await api.keywords.update(keyword.id, { active: !keyword.active }); await load(); }
    catch (caught) { setActionError(messageFromError(caught, "The phrase status could not be changed.")); }
  }

  async function deleteKeyword(keyword: Keyword) {
    if (!window.confirm(`Delete the phrase “${keyword.canonical_phrase}”? This cannot be undone.`)) return;
    setActionError("");
    try { await api.keywords.remove(keyword.id); setSuccess("Phrase deleted."); await load(); }
    catch (caught) { setActionError(messageFromError(caught, "The phrase could not be deleted.")); }
  }

  if (loading && !categories.length && !keywords.length) return <LoadingState label="Loading keywords" />;
  if (error) return <ErrorState message={error} onRetry={() => void load()} />;

  return (
    <>
      <div className="page-intro"><div><h2>Keyword categories and phrases</h2><p>Organize the Greek and English phrases that should be detected in operator speech.</p></div></div>
      {actionError ? <div className="form-error" role="alert">{actionError}</div> : null}
      {success ? <div className="success-message" role="status">{success}</div> : null}

      <div className="keyword-layout">
        <section aria-labelledby="categories-title">
          <div className="section-header"><div><h2 id="categories-title">Categories</h2><p>Group related phrases.</p></div></div>
          <form className="flat-panel stack-form" onSubmit={saveCategory}>
            <h3>{editingCategory !== undefined ? "Edit category" : "Create category"}</h3>
            <label><span>Name <span className="required">*</span></span><input value={categoryName} onChange={(event) => setCategoryName(event.target.value)} required /></label>
            <label><span>Description</span><textarea rows={3} value={categoryDescription} onChange={(event) => setCategoryDescription(event.target.value)} /></label>
            <div className="form-actions"><button className="button primary compact" type="submit" disabled={saving || !categoryName.trim()}>{saving ? "Saving…" : editingCategory !== undefined ? "Save category" : "Create category"}</button>{editingCategory !== undefined ? <button className="button secondary compact" type="button" onClick={resetCategory}>Cancel</button> : null}</div>
          </form>
          <div className="category-list" style={{ marginTop: 16 }}>
            {categories.length ? categories.map((category) => <div className="category-row" key={String(category.id)}><div className="category-row-header"><strong>{category.name}</strong><div className="inline-actions"><button className="text-button" type="button" onClick={() => beginCategoryEdit(category)}>Edit</button><button className="text-button" type="button" onClick={() => void deleteCategory(category)}>Delete</button></div></div>{category.description ? <p>{category.description}</p> : null}{typeof category.keyword_count === "number" ? <p>{category.keyword_count} phrase{category.keyword_count === 1 ? "" : "s"}</p> : null}</div>) : <p className="table-empty">No categories have been created.</p>}
          </div>
        </section>

        <section aria-labelledby="phrases-title">
          <div className="section-header"><div><h2 id="phrases-title">Phrases</h2><p>Alternative spellings can be entered one per line.</p></div></div>
          <form id="keyword-editor" className="flat-panel keyword-editor stack-form" onSubmit={saveKeyword}>
            <h3>{editingKeyword !== undefined ? "Edit phrase" : "Add phrase"}</h3>
            <div className="form-grid">
              <label><span>Category <span className="required">*</span></span><select value={draft.categoryId} onChange={(event) => setDraft((current) => ({ ...current, categoryId: event.target.value }))} required disabled={!categories.length}><option value="">Choose a category</option>{categories.map((category) => <option key={String(category.id)} value={String(category.id)}>{category.name}</option>)}</select></label>
              <label><span>Canonical phrase <span className="required">*</span></span><input value={draft.phrase} onChange={(event) => setDraft((current) => ({ ...current, phrase: event.target.value }))} required /></label>
              <label className="span-full"><span>Alternative spellings</span><textarea value={draft.variants} onChange={(event) => setDraft((current) => ({ ...current, variants: event.target.value }))} rows={3} /><span className="field-help">Enter each alternative on a separate line.</span></label>
              <label><span>Severity</span><select value={draft.severity} onChange={(event) => setDraft((current) => ({ ...current, severity: event.target.value }))}><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option><option value="critical">Critical</option></select></label>
              <label className="checkbox-row"><input type="checkbox" checked={draft.active} onChange={(event) => setDraft((current) => ({ ...current, active: event.target.checked }))} /><span>Phrase is active</span></label>
              <label className="span-full"><span>Notes</span><textarea value={draft.notes} onChange={(event) => setDraft((current) => ({ ...current, notes: event.target.value }))} rows={3} /></label>
            </div>
            <details className="optional-settings">
              <summary>Advanced matching settings</summary>
              <div className="stack-form">
                <label className="checkbox-row"><input type="checkbox" checked={draft.accentInsensitive} onChange={(event) => setDraft((current) => ({ ...current, accentInsensitive: event.target.checked }))} /><span>Ignore Greek accents</span></label>
                <label className="checkbox-row"><input type="checkbox" checked={draft.wholeWord} onChange={(event) => setDraft((current) => ({ ...current, wholeWord: event.target.checked }))} /><span>Match whole words only</span></label>
                <label className="checkbox-row"><input type="checkbox" checked={draft.exactPhrase} onChange={(event) => setDraft((current) => ({ ...current, exactPhrase: event.target.checked }))} /><span>Require the exact phrase</span></label>
                <label className="checkbox-row"><input type="checkbox" checked={draft.fuzzyMatch} onChange={(event) => setDraft((current) => ({ ...current, fuzzyMatch: event.target.checked }))} /><span>Allow close matches</span></label>
                {draft.fuzzyMatch ? <label><span>Close-match threshold</span><div className="threshold-row"><input type="range" min="0.5" max="1" step="0.01" value={draft.fuzzyThreshold} onChange={(event) => setDraft((current) => ({ ...current, fuzzyThreshold: Number(event.target.value) }))} /><output>{Math.round(draft.fuzzyThreshold * 100)}%</output></div><span className="field-help">Higher values require a closer match.</span></label> : null}
              </div>
            </details>
            <div className="form-actions"><button className="button primary" type="submit" disabled={saving || !categories.length || !draft.categoryId || !draft.phrase.trim()}>{saving ? "Saving…" : editingKeyword !== undefined ? "Save phrase" : "Add phrase"}</button>{editingKeyword !== undefined ? <button className="button secondary" type="button" onClick={resetKeyword}>Cancel</button> : null}</div>
          </form>

          <div className="table-wrap">
            <table><thead><tr><th scope="col"><SortButton label="Phrase" column="canonical_phrase" activeColumn={sortColumn} order={sortOrder} onSort={sort} /></th><th scope="col"><SortButton label="Category" column="category" activeColumn={sortColumn} order={sortOrder} onSort={sort} /></th><th scope="col">Severity</th><th scope="col"><SortButton label="Status" column="active" activeColumn={sortColumn} order={sortOrder} onSort={sort} /></th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>{sortedKeywords.length ? sortedKeywords.map((keyword) => <tr key={String(keyword.id)}><td>{keyword.canonical_phrase}{keyword.variants?.length ? <div className="subtle">{keyword.variants.length} alternative{keyword.variants.length === 1 ? "" : "s"}</div> : null}</td><td>{keyword.category_name || categories.find((category) => String(category.id) === String(keyword.category_id))?.name || "—"}</td><td>{keyword.severity}</td><td><StatusLabel status={keyword.active ? "enabled" : "disabled"} /></td><td className="actions"><button className="button secondary compact" type="button" onClick={() => void toggleKeyword(keyword)}>{keyword.active ? "Disable" : "Enable"}</button><button className="button secondary compact" type="button" onClick={() => beginKeywordEdit(keyword)}>Edit</button><button className="button danger compact" type="button" onClick={() => void deleteKeyword(keyword)}>Delete</button></td></tr>) : <TableEmpty colSpan={5}>No phrases have been added.</TableEmpty>}</tbody></table>
          </div>
        </section>
      </div>
    </>
  );
}
