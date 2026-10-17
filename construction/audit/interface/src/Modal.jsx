import { useEffect, useRef } from 'react';
import Icon from './Icon.jsx';

export default function Modal({ title, children, onClose, wide = false }) {
  const ref = useRef(null);
  useEffect(() => {
    const dialog = ref.current; dialog.showModal();
    const cancel = event => { event.preventDefault(); onClose?.(); };
    dialog.addEventListener('cancel', cancel);
    return () => { dialog.removeEventListener('cancel', cancel); dialog.close(); };
  }, [onClose]);
  return <dialog ref={ref} className={`modal ${wide ? 'modal-wide' : ''}`} aria-label={title}>
    <div className="modal-head"><h2>{title}</h2>{onClose && <button className="icon-button" onClick={onClose} aria-label="Close"><Icon name="close"/></button>}</div>
    {children}
  </dialog>;
}
