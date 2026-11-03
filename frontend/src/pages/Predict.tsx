import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { api, ApiError } from '../api'
import ConditionsForm from '../components/ConditionsForm'
import DataBadge from '../components/DataBadge'
import { ErrorBox, PageTitle, Skeleton, Spinner } from '../components/ui'
import { useFetch } from '../hooks/useFetch'
import { DEFAULT_FORM, fromInputs, lastInputs, toInputs, unusedKeys, validate, type Errors } from '../ranges'
import { useToast } from '../toast'
import type { FormState } from '../types'

export default function Predict() {
  const { t } = useTranslation()
  const toast = useToast()
  const nav = useNavigate()
  const meta = useFetch(api.crops)
  const [form, setForm] = useState<FormState>(() => { const l = lastInputs.load(); return l ? fromInputs(l) : DEFAULT_FORM })
  const [errors, setErrors] = useState<Errors>({})
  const [busy, setBusy] = useState(false)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    const v = validate(form, unusedKeys(meta.data?.input_schema?.used))
    setErrors(v)
    if (Object.keys(v).length) return toast('error', t('form.fix'))
    setBusy(true)
    try {
      const inputs = toInputs(form, meta.data?.input_schema?.used)
      const res = await api.predict(inputs)
      lastInputs.save(inputs)
      nav(`/result/${res.id}`)
    } catch (err) {
      toast('error', err instanceof ApiError ? (err.errors[0] ? `${err.errors[0].field}: ${err.errors[0].message}` : err.message) : t('common.error'))
    } finally { setBusy(false) }
  }

  return (
    <>
      <PageTitle title={t('predict.title')} sub={t('predict.subtitle')} />
      {meta.data && <div className="mb-5"><DataBadge mode={meta.data.data_mode} /></div>}
      {meta.error && <ErrorBox message={meta.error} onRetry={meta.reload} />}
      {meta.loading && <div className="space-y-4"><Skeleton className="h-40" /><Skeleton className="h-52" /><Skeleton className="h-52" /></div>}
      {meta.data && (
        <form onSubmit={submit} noValidate>
          <ConditionsForm form={form} setForm={setForm} errors={errors} meta={meta.data} />
          <div className="sticky bottom-0 -mx-4 mt-6 border-t border-line bg-bg/95 px-4 py-3 backdrop-blur">
            <button className="btn btn-primary w-full sm:w-auto sm:min-w-56" disabled={busy}>{busy && <Spinner />} {busy ? t('predict.running') : t('predict.submit')}</button>
          </div>
        </form>
      )}
    </>
  )
}